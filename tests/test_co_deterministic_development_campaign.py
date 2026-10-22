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


C=module('co_deterministic_campaign_test',ROOT/'scripts/co_deterministic_development_campaign.py')
H=module('co_deterministic_campaign_eval_test',ROOT/'scripts/evaluate_co_deterministic_dev.py')


def raw_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def semantic_receipt(full,staged,step,method):
    return {'protocol':'co-deterministic-semantic-state-comparison-v1',
            'status':'passed_exact_step5_prerestart' if step==5 else 'passed_exact_step10_continuation',
            'scope':'independent_prefix_readiness_not_resume_evidence' if step==5 else 'training_continuation_readiness_not_performance_evidence',
            'expected_step':step,'method':method,'compared_leaf_count':100,'differing_field_count':0,'differing_fields':[],
            'parameter_rng_values_exported':False,'raw_rollout_or_next_prompt_trace_exported':False,
            'live_loaded_optimizer_scheduler_rng_restoration_verified':False,
            'checkpoints':{'uninterrupted':full['complete_checkpoint'],'resumed':staged['complete_checkpoint']},
            'deterministic_runtime':{'uninterrupted':full['deterministic_runtime'],'resumed':staged['deterministic_runtime']}}


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

    def test_all_six_fixed_horizon_arms_and_eighteen_eval_endpoints_are_declared(self):
        root=Path('/synthetic/campaign');source=Path('/synthetic/management')
        plans=C.plans(root,source,'b'*64)
        self.assertEqual(len(plans),42);self.assertEqual(len({p['key'] for p in plans}),42)
        for task in C.TASKS:
            for method in C.METHODS:
                train=[p for p in plans if p['task']==task and p['method']==method and p['role']=='train']
                evaluation=[p for p in plans if p['task']==task and p['method']==method and p['role']=='eval']
                self.assertEqual([p['step'] for p in train],[100,200,300,400])
                self.assertEqual([p['step'] for p in evaluation],[100,200,400])
                for plan in train:
                    argv=plan['command'];self.assertEqual(argv[-1],str(C.TRAINING_SOURCE/'scripts/train_co_deterministic_dev_a100.slurm'))
                    self.assertIn('--time=02:00:00',argv);self.assertIn('--no-requeue',argv)
                    exports=next(x.removeprefix('--export=ALL,') for x in argv if x.startswith('--export='))
                    self.assertIn('CO_SOURCE_SHA='+C.TRAINING_SHA,exports)
                    self.assertIn('CO_DIAGNOSTIC=0',exports)
                    if plan['step']==100:self.assertIsNone(plan['depends_on'])
                    else:
                        self.assertEqual(plan['depends_on'],C.key('train',task,method,plan['step']-100))
                        self.assertIn(f'checkpoint-{plan["step"]-100}',exports)
                for plan in evaluation:
                    self.assertEqual(plan['command'][-1],str(source/'scripts/evaluate_co_deterministic_dev_a100.slurm'))
                    parent=C.key('train',task,method,plan['step'])
                    self.assertEqual(plan['depends_on'],parent)
                    resolved=C.resolved_command(plan,{parent:{'job_id':'991'}})
                    self.assertIn('CO_TRAIN_JOB_ID=991',next(x for x in resolved if x.startswith('--export=')))
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

    def test_parity_receipt_binds_both_current_states_and_applied_runtimes(self):
        for method in C.RESUME_METHODS:
            for step in (5,10):
                full={'complete_checkpoint':{'optimizer':'A'},'deterministic_runtime':{'proof':'full'}}
                staged={'complete_checkpoint':{'optimizer':'B'},'deterministic_runtime':{'proof':'staged'}}
                receipt=semantic_receipt(full,staged,step,method)
                C.validate_resume_equivalence(H,receipt,full,staged,step=step,method=method)
                for key,value in (('status','failed_state_difference'),('compared_leaf_count',0),('compared_leaf_count',True),
                                  ('differing_field_count',1),('differing_fields',['optimizer']),('expected_step',True),
                                  ('live_loaded_optimizer_scheduler_rng_restoration_verified',True),('deterministic_runtime',{})):
                    with self.subTest(method=method,step=step,key=key),self.assertRaises(ValueError):
                        C.validate_resume_equivalence(H,{**receipt,key:value},full,staged,step=step,method=method)
                changed=copy.deepcopy(staged);changed['deterministic_runtime']['proof']='another'
                with self.assertRaises(ValueError):C.validate_resume_equivalence(H,receipt,full,changed,step=step,method=method)

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

    def test_diagnostic_readiness_requires_both_continuations_and_all_eight_distinct_jobs(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(C,'DIAGNOSTICS',Path(directory)):
            root=Path(directory)
            proof={'initial':{name:{'job_id':str(100+i)} for i,name in enumerate(C.DIAGNOSTIC_CASES)},
                   'resume_commands':{method:['sbatch',method] for method in C.RESUME_METHODS}}
            with patch.object(C,'diagnostic_declaration',return_value=proof):
                def runner(argv):return argv[argv.index('-j')+1]+'|COMPLETED|0:0'
                self.assertFalse(C.diagnostic_readiness(H,runner)['scheduler_ready_for_full_evidence_validation'])
                for i,method in enumerate(C.RESUME_METHODS):
                    name=f'mis_{method}_staged-resume10'
                    raw_write(root/f'{name}-attempt.json',{'command':proof['resume_commands'][method]})
                    with self.assertRaisesRegex(ValueError,'unresolved'):C.diagnostic_readiness(H,runner)
                    raw_write(root/f'{name}-submission.json',{'returncode':0,'stdout':str(900+i)+'\n'})
                self.assertTrue(C.diagnostic_readiness(H,runner)['scheduler_ready_for_full_evidence_validation'])
                with self.assertRaisesRegex(ValueError,'diagnostic failed'):
                    C.diagnostic_readiness(H,lambda argv:argv[argv.index('-j')+1]+'|FAILED|1:0')
                raw_write(root/'mis_classic_grpo_staged-resume10-submission.json',{'returncode':0,'stdout':'900\n'})
                with self.assertRaisesRegex(ValueError,'reused'):C.diagnostic_readiness(H,runner)


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
        self.tick(submit=True,max_new_jobs=5)
        self.queue=[]
        self.tick(submit=True,max_new_jobs=1)
        # All other fresh arms must be admitted before the first eval100.
        evaluation=self.calls[-1]
        self.assertIn('CO_TRAIN_JOB_ID=100',next(x for x in evaluation if x.startswith('--export=')))
        self.assertEqual(evaluation[-1],str(self.source/'scripts/evaluate_co_deterministic_dev_a100.slurm'))

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
                latest[plan['method']]={'job_id':str(number),'complete_checkpoint':{'path':plan['train'],'step':400},'deterministic_runtime':{'applied':'bound'}}
        for method in C.METHODS:
            path=self.root/f'{method}-launch.json'
            H.write_once(path,{'training':latest[method]},'launch_sha256');launches[method]=path
        seen=[]
        def chunk(helper,protocol,task,method,step,train,job,argv,files,runner):
            seen.append((method,step))
            return {'job_id':job,'complete_checkpoint':{'path':str(train),'step':step},'deterministic_runtime':{'applied':'bound'}}
        def evaluation(helper,protocol,source,plan,submission,runner):
            return {'launch':H.identity(launches[plan['method']]),'job_id':submission['job_id']}
        before=set(self.root.iterdir())
        with patch.object(C,'attempts',return_value=submitted),patch.object(C,'validate_chunk',side_effect=chunk), \
             patch.object(C,'validate_evaluation',side_effect=evaluation):
            result=C.read_only_pair(self.root,'f'*64,'mis',400,runner=self.runner)
            self.assertEqual(seen,[(method,stop) for method in ('foarl_control','foarl_teacher') for stop in (100,200,300,400)])
            self.assertEqual([len(arm['training_chunks']) for arm in result['arms'].values()],[4,4])
            self.assertEqual(set(self.root.iterdir()),before)
            self.assertEqual(list((self.root/'terminals').iterdir()),[])
            seen.clear()
            classic=C.read_only_pair(self.root,'f'*64,'mis',400,baseline_method='classic_grpo',runner=self.runner)
            self.assertEqual(set(classic['arms']),{'classic_grpo','foarl_teacher'})
            self.assertEqual(seen,[(method,stop) for method in ('classic_grpo','foarl_teacher') for stop in (100,200,300,400)])
            self.assertEqual(set(self.root.iterdir()),before)
            wrong_runtime={**latest['foarl_teacher'],'deterministic_runtime':{'applied':'wrong'}}
            path=launches['foarl_teacher'];path.chmod(0o640);path.unlink()
            H.write_once(path,{'training':wrong_runtime},'launch_sha256')
            with self.assertRaisesRegex(ValueError,'actual declared training endpoint'):
                C.read_only_pair(self.root,'f'*64,'mis',400,runner=self.runner)
            wrong={**latest['foarl_teacher'],'job_id':'unrelated'}
            path=launches['foarl_teacher'];path.chmod(0o640);path.unlink()
            H.write_once(path,{'training':wrong},'launch_sha256')
            with self.assertRaisesRegex(ValueError,'actual declared training endpoint'):
                C.read_only_pair(self.root,'f'*64,'mis',400,runner=self.runner)


if __name__=='__main__':unittest.main()


class ExactDiagnosticGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.protocol={'management_source_manifest':{'path':str(self.root/'source/SOURCE_MANIFEST.sha256')},
                       'initialization':{'mis':{'checkpoint':'SFT','files':{}},'cvrp':{'checkpoint':'SFT-C','files':{}}},
                       'data_root':'/data','data_files':{'mis':{'fixed':'mis'},'cvrp':{'fixed':'cvrp'}},
                       'diagnostics':{'initial':{},'resume_commands':{}}}
        self.results={};self.prefixes={};self.resumed={}
        for i,(name,(task,method,step)) in enumerate(C.DIAGNOSTIC_CASES.items()):
            job=str(100+i);attempt=self.root/f'{name}-attempt.json'
            raw_write(attempt,{'command':['sbatch',name]})
            self.protocol['diagnostics']['initial'][name]={'job_id':job,'attempt':H.identity(attempt)}
            self.results[name]=self.state(name,step,job,step)
        for i,method in enumerate(C.RESUME_METHODS):
            full=f'mis_{method}_full';staged=f'mis_{method}_staged'
            self.prefixes[full]=self.state(full,5,self.results[full]['job_id'],10)
            self.prefixes[staged]=self.results[staged]
            prefix=semantic_receipt(self.prefixes[full],self.prefixes[staged],5,method)
            H.write_once(self.root/f'{method}-step5-prerestart-equivalence.json',prefix)
            command=['sbatch',method,'resume10'];self.protocol['diagnostics']['resume_commands'][method]=command
            job=str(900+i);self.resumed[staged]=self.state(staged,10,job,10)
            raw_write(self.root/f'{staged}-resume10-attempt.json',{'command':command})
            raw_write(self.root/f'{staged}-resume10-submission.json',{'returncode':0,'stdout':job+'\n'})
            (self.root/staged).mkdir()
            H.write_once(self.root/staged/'resume_equivalence.json',semantic_receipt(self.results[full],self.resumed[staged],10,method))
        self.root_patch=patch.object(C,'DIAGNOSTICS',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)

    @staticmethod
    def state(name,step,job,producer):
        return {'job_id':job,'complete_checkpoint':{'path':name,'step':step,'files':{'optimizer':'retained'}},
                'deterministic_runtime':{'applied_runtime':{'producer_stop':producer,'producer_job_id':job},
                                         'declaration_file':{'path':name},'applied_runtime_file':{'path':name+str(step)}}}

    def gate(self):
        def endpoint(helper,protocol,task,method,step,train,job,*args):
            result=(self.resumed[train.name] if step==10 and train.name.endswith('_staged') else self.results[train.name])
            self.assertEqual(job,result['job_id']);return result
        def checkpoint(train,step,contract,source):
            self.assertEqual(step,5);return self.prefixes[train.name]
        with patch.object(C,'validate_chunk',side_effect=endpoint),patch.object(H,'admit_checkpoint',side_effect=checkpoint):
            return C.diagnostic_gate(H,self.protocol,{},lambda _:self.fail('unexpected scheduler call'))

    def test_all_six_initial_two_prefixes_two_continuations_bound(self):
        result=self.gate()
        self.assertEqual(len(result['initial']),6)
        self.assertEqual(set(result['resumes']),set(C.RESUME_METHODS))
        self.assertEqual({r['resumed']['job_id'] for r in result['resumes'].values()},{'900','901'})
        self.assertEqual(result['status'],'all_six_diagnostics_two_exact_prefixes_and_two_continuations_verified')

    def test_bad_prefix_cannot_be_replaced_by_successful_step10_result(self):
        path=self.root/'classic_grpo-step5-prerestart-equivalence.json'
        original=H.read_json(path)
        for change in ('missing','failure','runtime','checkpoint'):
            value=copy.deepcopy(original)
            if change=='missing':path.unlink()
            else:
                if change=='failure':value['status']='failed_state_difference'
                elif change=='runtime':value['deterministic_runtime']['resumed']['applied_runtime']['producer_job_id']='999'
                else:value['checkpoints']['resumed']['files']['optimizer']='different'
                value['receipt_sha256']=H.digest({k:v for k,v in value.items() if k!='receipt_sha256'})
                path.chmod(0o640);raw_write(path,value)
            with self.subTest(change=change),self.assertRaises((ValueError,FileNotFoundError)):self.gate()
            raw_write(path,original)

    def test_both_continuations_must_succeed_and_bind_current_runtime(self):
        path=self.root/'mis_classic_grpo_staged/resume_equivalence.json'
        value=H.read_json(path);value['deterministic_runtime']['resumed']['applied_runtime']['producer_stop']=5
        value['receipt_sha256']=H.digest({k:v for k,v in value.items() if k!='receipt_sha256'})
        path.chmod(0o640);raw_write(path,value)
        with self.assertRaisesRegex(ValueError,'both current'):self.gate()

    def test_prefix_producer_must_be_its_actual_initial_allocation(self):
        self.prefixes['mis_classic_grpo_full']['deterministic_runtime']['applied_runtime']['producer_job_id']='999'
        with self.assertRaisesRegex(ValueError,'actual producer'):self.gate()

    def test_resume_cannot_reuse_an_initial_or_other_resume_job(self):
        path=self.root/'mis_classic_grpo_staged-resume10-submission.json'
        for job in ('100','900'):
            raw_write(path,{'returncode':0,'stdout':job+'\n'})
            with self.subTest(job=job),self.assertRaisesRegex(ValueError,'distinct job'):self.gate()


class NativeDeclarationBindingTest(unittest.TestCase):
    def test_fixed_six_plans_and_actual_submissions_are_pinned_without_importing_untrusted_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';native=root/'diagnostics'
            (source/'scripts').mkdir(parents=True);native.mkdir()
            producer=source/'scripts/co_deterministic_diagnostics.py'
            producer.write_bytes((ROOT/'scripts/co_deterministic_diagnostics.py').read_bytes())
            diagnostic=module('co_producer_synthetic',producer)
            declaration=diagnostic.protocol_material(source,'a'*64,native)
            raw_write(native/'protocol.json',declaration)
            for i,plan in enumerate(declaration['plans']):
                raw_write(native/f'{plan["key"]}-attempt.json',{'command':plan['command']})
                raw_write(native/f'{plan["key"]}-submission.json',{'returncode':0,'stdout':str(100+i)+'\n'})
            sha=H.identity(native/'protocol.json')['sha256']
            with patch.object(C,'DIAGNOSTICS',native),patch.object(C,'TRAINING_SOURCE',source), \
                 patch.object(C,'TRAINING_SHA','a'*64),patch.object(C,'DIAGNOSTIC_PROTOCOL_SHA',sha):
                actual=C.diagnostic_declaration(H)
                self.assertEqual(len(actual['initial']),6)
                self.assertEqual(len(actual['resume_commands']),2)
                raw_write(native/'protocol.json',{**declaration,'fixed_horizon':10})
                with patch.object(C,'load_module') as unsafe:
                    with self.assertRaisesRegex(ValueError,'externally pinned'):C.diagnostic_declaration(H)
                    unsafe.assert_not_called()

    def test_classic_contract_preserves_exact_beta_zero_schema(self):
        protocol={'initialization':{'mis':{'checkpoint':'SFT','files':{}}},'data_root':'/data','data_files':{'mis':{}}}
        result=C.expected_contract(protocol,'mis','classic_grpo')
        self.assertEqual(result['beta'],0.)
        self.assertEqual(result['teacher_weight'],0.)
        other=C.expected_contract(protocol,'mis','foarl_control')
        self.assertNotIn('beta',other)
        self.assertEqual({k:v for k,v in result.items() if k not in ('beta','method')},
                         {k:v for k,v in other.items() if k!='method'})


# The actual common launcher fixture synthesizes machine boundaries only; its
# signed runtime declarations, snapshots and four-rank proofs are real files.
import pytest

LAUNCH = module('co_controller_launcher_fixture',ROOT/'tests/test_co_deterministic_launcher.py')


@pytest.mark.parametrize('method',['classic_grpo','foarl_control','foarl_teacher'])
@pytest.mark.parametrize('diagnostic,step',[(False,100),(True,10)])
def test_actual_new_launcher_receipts_pass_controller_endpoint_admission(tmp_path,monkeypatch,method,diagnostic,step):
    fixture=LAUNCH.chunk.__wrapped__(tmp_path,monkeypatch)
    import os
    import shlex
    os.environ.update(CO_METHOD=method,CO_DIAGNOSTIC=str(int(diagnostic)),CO_STOP=str(step))
    fixture.module.main()
    train=fixture.output;m=fixture.module
    contract=m.read_json(train/'local_development_run_contract.json')
    complete_path=train/f'chunk-{step}-complete.json'
    complete=H.read_json(complete_path);complete['receipt_sha256']=H.digest(complete)
    complete_path.chmod(0o640);raw_write(complete_path,complete);complete_path.chmod(0o440)
    runtime={'declaration':m.read_only_runtime_declaration(train,contract),
             'declaration_file':H.identity(train/m.DECLARATION_NAME),
             'applied_runtime':m.read_only_runtime_evidence(train/'checkpoints'/f'checkpoint-{step}',contract),
             'applied_runtime_file':H.identity(train/'checkpoints'/f'checkpoint-{step}'/m.EVIDENCE_NAME)}
    admitted={'complete_checkpoint':complete,'deterministic_runtime':runtime}
    monkeypatch.setattr(H,'admit_checkpoint',lambda *a:admitted)
    monkeypatch.setattr(H,'RUNTIME',m.RUNTIME);monkeypatch.setattr(H,'TRAIN_DATA',fixture.data)
    monkeypatch.setattr(C,'TRAINING_SOURCE',fixture.source);monkeypatch.setattr(C,'TRAINING_SHA',contract['source']['manifest_sha256'])
    protocol={'initialization':{'mis':contract['initialization']},'data_root':contract['data_root'],
              'data_files':{'mis':contract['data_files']},
              'management_source_manifest':{'path':str(fixture.source/'SOURCE_MANIFEST.sha256')}}
    stdout=train/'job-12345.out';stderr=train/'job-12345.err'
    stdout.write_text(f'CO_DETERMINISTIC_CHUNK_COMPLETE task=mis method={method} step={step}\n')
    stderr.write_text('')
    argv=['sbatch','--parsable','--job-name=exact-producer','--output='+str(train/'job-%j.out'),
          '--error='+str(train/'job-%j.err'),str(fixture.source/'scripts/train_co_deterministic_dev_a100.slurm')]
    def runner(command):
        if 'State,ExitCode' in command[-1]:
            return '12345|COMPLETED|0:0|1800|120|cpu=80,billing=5,gres/gpu=4|qos_gpu_a100-dev|vas@a100|gpu_p5'
        return '|'.join(('12345','exact-producer',str(fixture.source),shlex.join(argv),str(train/'job-%j.out'),str(train/'job-%j.err')))
    result=C.validate_chunk(H,protocol,'mis',method,step,train,'12345',argv,{str(i):'fixed' for i in range(19)},runner,diagnostic)
    assert result['allocation']['allocated_gpu_hours']==2.
    assert result['deterministic_runtime']==runtime
    assert result['complete_checkpoint']==complete
    for change in ('job','stop','marker','runtime'):
        if change=='job':
            with pytest.raises(ValueError):C.validate_chunk(H,protocol,'mis',method,step,train,'999',argv,{str(i):'fixed' for i in range(19)},runner,diagnostic)
        elif change=='stop':
            with pytest.raises(ValueError):C.validate_chunk(H,protocol,'mis',method,step,train,'12345',argv,{},runner,diagnostic)
        elif change=='marker':
            stdout.write_text('CO_DETERMINISTIC_CHUNK_COMPLETE task=cvrp method='+method+f' step={step}\n')
            with pytest.raises(ValueError):C.validate_chunk(H,protocol,'mis',method,step,train,'12345',argv,{str(i):'fixed' for i in range(19)},runner,diagnostic)
            stdout.write_text(f'CO_DETERMINISTIC_CHUNK_COMPLETE task=mis method={method} step={step}\n')
        else:
            runtime['applied_runtime']['producer_job_id']='999'
            with pytest.raises(ValueError,match='another training job'):C.validate_chunk(H,protocol,'mis',method,step,train,'12345',argv,{str(i):'fixed' for i in range(19)},runner,diagnostic)


def test_refresh_crossbinds_evaluation_applied_runtime_even_when_checkpoint_and_job_match(tmp_path,monkeypatch):
    root=tmp_path; (root/'terminals').mkdir()
    plans=C.plans(root,root/'source','a'*64)
    train=next(p for p in plans if p['role']=='train' and p['method']=='classic_grpo')
    evaluation=next(p for p in plans if p['role']=='eval' and p['depends_on']==train['key'])
    protocol={'protocol_sha256':'p','plans':[train,evaluation]}
    submission={p['key']:{'job_id':str(i),'attempt':{'command':p['command']},'attempt_identity':{},'submission':{}}
                for i,p in enumerate(protocol['plans'],100)}
    actual={'job_id':'100','complete_checkpoint':{'files':'same'},'deterministic_runtime':{'applied':'correct'}}
    launch=root/'launch.json';H.write_once(launch,{'training':actual},'launch_sha256')
    monkeypatch.setattr(C,'validate_chunk',lambda *a:actual)
    monkeypatch.setattr(C,'validate_evaluation',lambda *a:{'launch':H.identity(launch),'job_id':'101'})
    monkeypatch.setattr(C,'job_state',lambda job,runner:{'job_id':job,'state':'COMPLETED','exit_code':'0:0'})
    _,verified=C.refresh(H,root,protocol,root,{},submission,None)
    assert len(verified)==2 and not list((root/'terminals').iterdir())
    launch.chmod(0o640);launch.unlink()
    H.write_once(launch,{'training':{**actual,'deterministic_runtime':{'applied':'wrong'}}},'launch_sha256')
    with pytest.raises(ValueError,match='actual declared training job'):
        C.refresh(H,root,protocol,root,{},submission,None)
    assert not list((root/'terminals').iterdir())


def test_management_may_add_but_cannot_change_any_of_original_training_inputs(tmp_path,monkeypatch):
    source=tmp_path/'management';source.mkdir();(source/'logs').mkdir()
    producers=('scripts/run_co_deterministic_dev_chunk.py','scripts/train_co_deterministic_dev_a100.slurm',
               'scripts/compare_co_deterministic_state.py','scripts/co_deterministic_diagnostics.py')
    original={**{name:'original' for name in producers},'tests/frozen_old_regression.py':'retain-even-test-bytes'}
    files={**original,C.SELF:'self',C.EVALUATOR:'new'}
    def source_reader(root,sha,**kwargs):
        return ({'sha256':sha},files if kwargs.get('management') else original)
    helper=SimpleNamespace(validate_source=source_reader,identity=lambda path:{'sha256':'self'},
                           validate_scientific_equivalence=lambda *a:None)
    monkeypatch.setattr(C,'bootstrap_source',lambda *a:None)
    monkeypatch.setattr(C,'evaluation_module',lambda *a:helper)
    assert C.check_source(source,'a'*64)[-1]==original
    files['tests/frozen_old_regression.py']='changed'
    with pytest.raises(ValueError,match='original frozen training snapshot input'):
        C.check_source(source,'a'*64)
