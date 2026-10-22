"""Model-free declaration, dependency, gate and one-attempt scheduler tests."""
import copy
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


C=module('foarl_teacher_campaign_test',ROOT/'scripts/foarl_teacher_development_campaign.py')
H=module('foarl_teacher_campaign_eval_test',ROOT/'scripts/evaluate_foarl_teacher_dev.py')


def raw_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class DeclarationTest(unittest.TestCase):
    def test_untrusted_evaluator_is_rejected_before_it_can_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir()
            (root/C.SELF).write_bytes((ROOT/C.SELF).read_bytes())
            helper=root/C.EVALUATOR;helper.write_text('raise RuntimeError("original synthetic helper")\n')
            manifest=root/'SOURCE_MANIFEST.sha256'
            manifest.write_text(''.join(hashlib.sha256((root/name).read_bytes()).hexdigest()+'  '+name+'\n'
                                        for name in (C.SELF,C.EVALUATOR)))
            sha=hashlib.sha256(manifest.read_bytes()).hexdigest();sentinel=root/'executed-untrusted-code'
            helper.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').touch()\n')
            with self.assertRaisesRegex(ValueError,'bootstrap executable bytes'):C.check_source(root,sha)
            self.assertFalse(sentinel.exists())

    def test_protocol_file_pin_precedes_any_source_selection_or_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw_write(root/'protocol.json',{'management_source_manifest':{'path':'/untrusted/source/SOURCE_MANIFEST.sha256','sha256':'a'*64}})
            with patch.object(C,'check_source') as importer:
                with self.assertRaisesRegex(ValueError,'before source selection'):C.load_protocol(root,'b'*64)
                importer.assert_not_called()

    def test_all_four_fixed_horizon_arms_and_twelve_eval_endpoints_are_declared(self):
        root=Path('/synthetic/campaign');source=Path('/synthetic/management')
        plans=C.plans(root,source,'b'*64)
        self.assertEqual(len(plans),28);self.assertEqual(len({p['key'] for p in plans}),28)
        for task in C.TASKS:
            for method in C.METHODS:
                train=[p for p in plans if p['task']==task and p['method']==method and p['role']=='train']
                evaluation=[p for p in plans if p['task']==task and p['method']==method and p['role']=='eval']
                self.assertEqual([p['step'] for p in train],[100,200,300,400])
                self.assertEqual([p['step'] for p in evaluation],[100,200,400])
                for plan in train:
                    argv=plan['command'];self.assertEqual(argv[-1],str(C.TRAINING_SOURCE/'scripts/train_foarl_teacher_dev_a100.slurm'))
                    self.assertIn('--time=02:00:00',argv);self.assertIn('--no-requeue',argv)
                    exports=next(x.removeprefix('--export=ALL,') for x in argv if x.startswith('--export='))
                    self.assertIn('FT_SOURCE_SHA='+C.TRAINING_SHA,exports)
                    self.assertIn('FT_DIAGNOSTIC=0',exports)
                    if plan['step']==100:self.assertIsNone(plan['depends_on'])
                    else:
                        self.assertEqual(plan['depends_on'],C.key('train',task,method,plan['step']-100))
                        self.assertIn(f'checkpoint-{plan["step"]-100}',exports)
                for plan in evaluation:
                    self.assertEqual(plan['command'][-1],str(source/'scripts/evaluate_foarl_teacher_dev_a100.slurm'))
                    parent=C.key('train',task,method,plan['step'])
                    self.assertEqual(plan['depends_on'],parent)
                    resolved=C.resolved_command(plan,{parent:{'job_id':'991'}})
                    self.assertIn('FT_TRAIN_JOB_ID=991',next(x for x in resolved if x.startswith('--export=')))
                    self.assertNotEqual(plan['command'],resolved)

    def test_evaluation_cannot_resolve_without_its_actual_training_job(self):
        plan=next(x for x in C.plans(Path('/r'),Path('/s'),'a'*64) if x['role']=='eval')
        with self.assertRaises(ValueError):C.resolved_command(plan,{})

    def test_contract_pins_original_source_init_data_horizon_and_both_teacher_weights(self):
        protocol={'initialization':{'mis':{'checkpoint':'original-SFT','files':{}}},'data_root':'/data',
                  'data_files':{'mis':{'mis/train/train.json':{'sha256':'data'}}}}
        control=C.expected_contract(protocol,'mis','foarl_control')
        teacher=C.expected_contract(protocol,'mis','foarl_teacher')
        self.assertEqual(control['source'],{'root':str(C.TRAINING_SOURCE),'manifest_sha256':C.TRAINING_SHA})
        self.assertEqual(control['fixed_horizon'],{'max_steps':400,'world_size':4})
        self.assertEqual(control['initialization'],teacher['initialization'])
        self.assertEqual(control['data_files'],teacher['data_files'])
        self.assertEqual(control['teacher_weight'],0.);self.assertEqual(teacher['teacher_weight'],.05)
        self.assertEqual(C.expected_contract(protocol,'mis','foarl_teacher',True)['schedule']['save_steps'],5)

    def test_parity_receipt_must_pass_and_bind_both_current_full_states(self):
        full={'complete_checkpoint':{'files':{'optimizer.pt':'A'},'source':'frozen'}}
        staged={'complete_checkpoint':{'files':{'optimizer.pt':'B'},'source':'frozen'}}
        receipt={'protocol':'foarl-teacher-exact-resume-comparison-v1','status':'passed_exact_step10_continuation',
                 'scope':'training_continuation_readiness_not_performance_evidence','compared_leaf_count':100,
                 'differing_field_count':0,'differing_fields':[],'parameter_rng_values_exported':False,
                 'checkpoints':{'uninterrupted':full['complete_checkpoint'],'resumed':staged['complete_checkpoint']}}
        C.validate_resume_equivalence(H,receipt,full,staged)
        for key,value in (('status','failed_state_difference'),('compared_leaf_count',0),('compared_leaf_count',True),
                          ('differing_field_count',1),('differing_fields',['optimizer']),('checkpoints',{'uninterrupted':full['complete_checkpoint']})):
            with self.subTest(key=key),self.assertRaises(ValueError):C.validate_resume_equivalence(H,{**receipt,key:value},full,staged)
        altered=copy.deepcopy(staged);altered['complete_checkpoint']['files']['optimizer.pt']='changed'
        with self.assertRaises(ValueError):C.validate_resume_equivalence(H,receipt,full,altered)

    def test_actual_terminal_resource_cost_distinguishes_billing_from_reserved_gpu(self):
        raw='123|COMPLETED|0:0|3600|120|cpu=80,billing=5,gres/gpu=4,mem=300000M|qos_gpu_a100-dev|vas@a100|gpu_p5'
        result=C.accounting('123',4,lambda _:raw)
        self.assertEqual(result['allocated_gpu_hours'],4)
        self.assertEqual(result['allocated_cpu_hours'],80)
        self.assertEqual(result['billing_unit_hours'],5)
        for bad in (raw.replace('COMPLETED','FAILED'),raw.replace('0:0','1:0'),raw.replace('cpu=80','cpu=80,cpu=40'),
                    raw.replace('gres/gpu=4','gres/gpu=1'),raw.replace('|120|','|25|')):
            with self.assertRaises(ValueError):C.accounting('123',4,lambda _:bad)

    def test_submission_response_rejects_failure_multiline_and_ambiguous_ids(self):
        self.assertEqual(C.accepted_job({'returncode':0,'stdout':'123;cluster\n'}),'123')
        for code,text in ((1,'123'),(0,''),(0,'123\n124'),(0,'123_1'),(0,'not-a-job'),(False,'123')):
            with self.subTest(code=code,text=text),self.assertRaises(ValueError):C.accepted_job({'returncode':code,'stdout':text})


class QueueAndReadinessTest(unittest.TestCase):
    def test_unknown_and_absent_accounting_remain_reserved(self):
        submitted={'a':{'job_id':'123'},'b':{'job_id':'124'},'c':{'job_id':'125'}}
        def runner(argv):
            if argv[0]=='squeue':return '1\n2\n'
            job=argv[argv.index('-j')+1]
            return {'123':'','124':'124|CONFIGURING|0:0','125':'125|COMPLETED|0:0'}[job]
        result=C.occupancy(submitted,runner)
        self.assertEqual(result['reserved_job_slots'],4)
        self.assertEqual(result['accepted_but_not_visible'],['123','124'])

    def test_duplicate_and_unexpanded_array_queue_are_rejected(self):
        for raw in ('1\n1\n','12_[1-5]\n','abc\n'):
            with self.assertRaises(ValueError):C.occupancy({},lambda _:raw)
        self.assertEqual(C.occupancy({},lambda _:'12_1\n12_2\n')['reserved_job_slots'],2)

    def test_diagnostic_readiness_waits_for_resume_and_rejects_failure(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(C,'DIAGNOSTICS',Path(directory)):
            def runner(argv):return argv[argv.index('-j')+1]+'|COMPLETED|0:0'
            self.assertFalse(C.diagnostic_readiness(H,runner)['scheduler_ready_for_full_evidence_validation'])
            root=Path(directory)
            raw_write(root/'mis_teacher_staged-resume10-attempt.json',{'command':[]})
            with self.assertRaisesRegex(ValueError,'unresolved'):C.diagnostic_readiness(H,runner)
            raw_write(root/'mis_teacher_staged-resume10-submission.json',{'returncode':0,'stdout':'999\n'})
            self.assertTrue(C.diagnostic_readiness(H,runner)['scheduler_ready_for_full_evidence_validation'])
            with self.assertRaisesRegex(ValueError,'diagnostic failed'):
                C.diagnostic_readiness(H,lambda argv:argv[argv.index('-j')+1]+'|FAILED|1:0')


class TickTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        (self.root/'.submission.lock').touch()
        (self.root/'terminals').mkdir();(self.root/'runs').mkdir();(self.root/'evaluations').mkdir()
        self.source=self.root/'source'
        self.protocol={'protocol_sha256':'p'*64,'plans':C.plans(self.root,self.source,'a'*64)}
        self.queue=[str(x) for x in range(1,7)];self.states={};self.calls=[]
        self.ready={'scheduler_ready_for_full_evidence_validation':True,'jobs':{}}
        self.load=patch.object(C,'load_protocol',return_value=(H,self.protocol,self.source,{})).start()
        self.refresh=patch.object(C,'refresh',return_value=({},{})).start()
        self.readiness=patch.object(C,'diagnostic_readiness',return_value=self.ready).start()
        self.gate=patch.object(C,'diagnostic_gate',return_value={'status':'gate_passed','fixed':'evidence'}).start()
        self.addCleanup(patch.stopall)

    def runner(self,argv):
        if argv[0]=='squeue':return '\n'.join(self.queue)
        job=argv[argv.index('-j')+1]
        return job+'|'+self.states.get(job,'PENDING')+'|0:0'

    def submitter(self,argv):
        # The same physical lock must remain held across scheduler mutation.
        with (self.root/'.submission.lock').open('a') as other:
            with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=str(100+len(self.calls));self.calls.append(argv);self.states[job]='PENDING'
        return SimpleNamespace(returncode=0,stdout=job+'\n',stderr='')

    def tick(self,**kwargs):
        return C.tick(self.root,'f'*64,runner=self.runner,submitter=self.submitter,**kwargs)

    def test_bounded_four_submissions_with_actual_lock_then_no_duplicate_attempt(self):
        result=self.tick(submit=True,max_new_jobs=4)
        self.assertEqual(len(result['submitted_this_tick']),4);self.assertEqual(len(self.calls),4)
        self.assertTrue(all('train_' in item['key'] and item['key'].endswith('s100') for item in result['submitted_this_tick']))
        self.assertEqual(len(list(self.root.glob('*-attempt.json'))),4)
        again=self.tick(submit=True,max_new_jobs=4)
        self.assertEqual(again['submitted_this_tick'],[]);self.assertEqual(len(self.calls),4)

    def test_nine_slots_admits_one_invisible_job_ten_admits_none(self):
        self.queue=[str(x) for x in range(1,10)]
        result=self.tick(submit=True)
        self.assertEqual(len(result['submitted_this_tick']),1)
        self.assertEqual(result['queue']['reserved_job_slots'],10)
        again=self.tick(submit=True);self.assertEqual(again['submitted_this_tick'],[])

    def test_status_has_no_gate_writes_or_submissions(self):
        result=self.tick(submit=False)
        self.assertEqual(result['status'],'inspected_declared_campaign');self.assertEqual(self.calls,[])
        self.assertFalse((self.root/'diagnostic_gate.json').exists());self.gate.assert_not_called()

    def test_pending_or_failed_gate_never_consumes_a_production_attempt(self):
        self.readiness.return_value={'scheduler_ready_for_full_evidence_validation':False,'jobs':{}}
        result=self.tick(submit=True)
        self.assertEqual(result['status'],'waiting_for_declared_diagnostics_and_resume')
        self.assertEqual(self.calls,[]);self.assertEqual(list(self.root.glob('*-attempt.json')),[])
        self.readiness.return_value=self.ready;self.gate.side_effect=ValueError('native parity failed')
        with self.assertRaises(ValueError):self.tick(submit=True)
        self.assertEqual(list(self.root.glob('*-attempt.json')),[])

    def test_timeout_leaves_durable_attempt_and_blocks_all_further_mutations(self):
        def timeout(argv):raise subprocess.TimeoutExpired(argv,60)
        with self.assertRaises(subprocess.TimeoutExpired):C.tick(self.root,'f'*64,submit=True,runner=self.runner,submitter=timeout)
        self.assertEqual(len(list(self.root.glob('*-attempt.json'))),1)
        self.assertEqual(list(self.root.glob('*-submission.json')),[])
        with self.assertRaisesRegex(ValueError,'unresolved'):self.tick(submit=True)
        self.assertEqual(self.calls,[])

    def test_rejected_scheduler_attempt_cannot_be_automatically_retried(self):
        def reject(argv):return SimpleNamespace(returncode=1,stdout='',stderr='scheduler refusal')
        with self.assertRaises(ValueError):C.tick(self.root,'f'*64,submit=True,runner=self.runner,submitter=reject)
        with self.assertRaises(ValueError):self.tick(submit=True)
        self.assertEqual(len(list(self.root.glob('*-attempt.json'))),1);self.assertEqual(self.calls,[])

    def test_eval_substitution_and_next_chunk_require_verified_parent(self):
        parent=self.protocol['plans'][0]
        self.tick(submit=True,max_new_jobs=1)
        submitted=C.attempts(H,self.root,self.protocol)
        self.refresh.return_value=({}, {parent['key']:{'job_id':submitted[parent['key']]['job_id']}})
        # Select only the predeclared evaluation and next training chunk after
        # showing that no other arm may bypass its own dependency.
        targets=[p for p in self.protocol['plans'] if p['depends_on']==parent['key']]
        self.assertEqual({p['role'] for p in targets},{'eval','train'})
        self.queue=[]
        self.tick(submit=True,max_new_jobs=4)
        # Other three fresh arms come first; the fourth new command is eval100.
        evaluation=self.calls[-1]
        self.assertIn('FT_TRAIN_JOB_ID=100',next(x for x in evaluation if x.startswith('--export=')))
        self.assertEqual(evaluation[-1],str(self.source/'scripts/evaluate_foarl_teacher_dev_a100.slurm'))

    def test_existing_unsubmitted_output_refuses_hidden_retry(self):
        first=self.protocol['plans'][0]
        Path(first['output']).mkdir()
        with self.assertRaisesRegex(ValueError,'hidden retry'):self.tick(submit=True)
        self.assertEqual(self.calls,[])

    def test_changed_attempt_and_unknown_key_are_rejected(self):
        self.tick(submit=True,max_new_jobs=1)
        path=next(self.root.glob('*-attempt.json'));record=H.read_json(path,'attempt_sha256')
        record['command'][-1]='/another/launcher'
        record['attempt_sha256']=H.digest({k:v for k,v in record.items() if k!='attempt_sha256'})
        path.chmod(0o640);raw_write(path,record)
        with self.assertRaises(ValueError):self.tick(submit=True)

    def test_orphan_submission_or_terminal_blocks_an_unrecorded_repeat(self):
        name=self.protocol['plans'][0]['key']
        orphan=self.root/f'{name}-submission.json';raw_write(orphan,{'stdout':'123','returncode':0})
        with self.assertRaisesRegex(ValueError,'orphan'):self.tick(submit=True)
        orphan.unlink();raw_write(self.root/'terminals'/f'{name}.json',{'job_id':'123'})
        with self.assertRaisesRegex(ValueError,'no retained submission'):self.tick(submit=True)
        self.assertEqual(self.calls,[])

    def test_competing_controller_cannot_enter_same_campaign_lock(self):
        with (self.root/'.submission.lock').open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.tick(submit=True)
        self.assertEqual(self.calls,[])

    def test_read_only_pair_revalidates_all_cumulative_chunks_without_writing_terminals(self):
        (self.root/'.submission.lock').touch();raw_write(self.root/'protocol.json',self.protocol)
        H.write_once(self.root/'diagnostic_gate.json',self.gate.return_value,'gate_sha256')
        self.protocol['benchmark']={'fixed':'dev100'}
        self.protocol['management_source_manifest']={'path':str(self.source/'SOURCE_MANIFEST.sha256'),'sha256':'a'*64}
        submitted={};latest={};launches={}
        for number,plan in enumerate(self.protocol['plans'],start=700):
            submitted[plan['key']]={'job_id':str(number),'attempt':{'command':plan['command']}}
            if plan['role']=='train' and plan['step']==400:
                latest[plan['method']]={'job_id':str(number),'complete_checkpoint':{'path':plan['train'],'step':400}}
        for method in C.METHODS:
            path=self.root/f'{method}-launch.json'
            H.write_once(path,{'training':latest[method]},'launch_sha256');launches[method]=path
        seen=[]
        def chunk(helper,protocol,task,method,step,train,job,argv,files,runner):
            seen.append((method,step))
            return {'job_id':job,'complete_checkpoint':{'path':str(train),'step':step}}
        def evaluation(helper,protocol,source,plan,submission,runner):
            return {'launch':H.identity(launches[plan['method']]),'job_id':submission['job_id']}
        before=set(self.root.iterdir())
        with patch.object(C,'attempts',return_value=submitted),patch.object(C,'validate_chunk',side_effect=chunk), \
             patch.object(C,'validate_evaluation',side_effect=evaluation):
            result=C.read_only_pair(self.root,'f'*64,'mis',400,runner=self.runner)
            self.assertEqual(seen,[(method,stop) for method in C.METHODS for stop in (100,200,300,400)])
            self.assertEqual([len(arm['training_chunks']) for arm in result['arms'].values()],[4,4])
            self.assertEqual(set(self.root.iterdir()),before)
            self.assertEqual(list((self.root/'terminals').iterdir()),[])
            wrong={**latest['foarl_teacher'],'job_id':'unrelated'}
            path=launches['foarl_teacher'];path.chmod(0o640);path.unlink()
            H.write_once(path,{'training':wrong},'launch_sha256')
            with self.assertRaisesRegex(ValueError,'actual declared training endpoint'):
                C.read_only_pair(self.root,'f'*64,'mis',400,runner=self.runner)


if __name__=='__main__':unittest.main()
