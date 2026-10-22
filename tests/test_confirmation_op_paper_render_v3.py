"""Synthetic complete matrices only; no policy outputs or actor imports."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'experiments/rloo_one_shot_20261016/render_confirmation_op_paper_v3.py'

def module(path=PATH):
    raw=path.read_bytes();m=types.ModuleType('confirmation_renderer_test')
    m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
    exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m

@pytest.fixture(scope='module',params=['pure','teacher'])
def matrix(request):
    R=module();A=module(PATH.parent/'analyze_confirmation_op_v3.py')
    methods,candidate,controls=A.recipe(request.param)
    cohort=[{'instance_id':f'synthetic{i}','semantic_id':f'fixturephysical{i}',
        'stratum':0 if i<86 else (1 if i<171 else 2),'total_prize':100,'reference_objective':50} for i in range(256)]
    def rows(prize):
        return [{'instance_id':r['instance_id'],'reference_id':r['semantic_id'],'feasible':True,'objective':prize} for r in cohort]
    trained=[{'seed':s,'methods':{m:rows(90 if m==candidate else 70) for m in methods}} for s in R.SEEDS]
    stats=A.compute_statistics(branch=request.param,cohort=cohort,trained_by_seed=trained,sft_once={'seed':101,'arm':'sft','records':rows(65)})
    return {'synthetic_fixture':True,'status':'synthetic_confirmation_OP_layout_fixture','branch':request.param,
        'metadata_admitted':True,'statistics':stats,'registered_global_IUT_passed':True,'scoped_confirmation_established':True,
        'superiority_established':False,'publication_eligible':False}

def test_default_no_files(tmp_path):
    r=subprocess.run([sys.executable,'-I','-B',str(PATH)],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)['status']=='prepared_only_no_paper_changes'
    assert list(tmp_path.iterdir())==[]

def test_actual_full_statistics_schema_renders_both_branches(matrix):
    R=module();texts=R._layout(matrix,synthetic_fixture=True)
    methods=R.PURE if matrix['branch']=='pure' else R.TEACHER
    own=texts[R.NAMES[0]];pairs=texts[R.NAMES[1]];section=texts[R.NAMES[2]]
    assert len(texts)==3 and all('SYNTHETIC LAYOUT FIXTURE; NO MODEL RESULTS.' in s for s in texts.values())
    assert all(R.LABELS[m] in own for m in ('sft',*methods))
    assert all(R.LABELS[m] in pairs for m in matrix['statistics']['controls'])
    assert '1280/1280' in own and '256/256' in own and '95.00' in own
    assert 'not certified optima' in own and 'negative gaps improve' in section
    assert '$p=1/32$' in pairs and 'PASSED' in pairs and '5/5/5' in pairs
    assert 'population-mean superiority' in section and 'no observed' in section
    assert 'not five independent SFT measurements' in section

def test_failed_gate_remains_reportable_without_win(matrix):
    R=module();x=copy.deepcopy(matrix);stats=x['statistics'];control=stats['controls'][0]
    # A fourth favorable utility sign, with the tie retained in the five-seed
    # denominator, makes this conjunctive claim fail at p=6/32=3/16.
    stats['comparisons'][control]['per_seed'][0]['exact_effects']['utility']={'numerator':'0','denominator':'1'}
    g=stats['global_IUT'];c=g['component_tests'][control]['utility']
    c.update(favorable_by_seed=[False,True,True,True,True],favorable_seeds=4,p_exact={'numerator':'3','denominator':'16'},p_value=3/16)
    g.update(global_p_exact={'numerator':'3','denominator':'16'},global_p_value=3/16,statistical_gate_passed=False)
    x.update(registered_global_IUT_passed=False,scoped_confirmation_established=False)
    texts=R._layout(x,synthetic_fixture=True)
    assert 'NOT PASSED' in texts[R.NAMES[1]] and '$p=3/16$' in texts[R.NAMES[1]]
    assert 'did not pass' in texts[R.NAMES[2]] and 'does not establish' in texts[R.NAMES[2]]

@pytest.mark.parametrize('change',[
    lambda x:x['statistics']['summaries'].pop('503'),
    lambda x:x['statistics']['comparisons'].pop('sft'),
    lambda x:x.update(superiority_established=True),
    lambda x:x.update(registered_global_IUT_passed=False),
    lambda x:x['statistics']['graph_bootstrap'].update(seed_resampling=True),
    lambda x:x['statistics']['graph_bootstrap'].update(replicates=100),
    lambda x:x['statistics']['summaries']['101']['sft'].update(n_feasible=True),
    lambda x:x['statistics']['summaries']['211']['sft'].update(mean_U10=.5),
    lambda x:x['statistics']['summaries']['101']['rloo_cost'].update(mean_Uphys=float('nan')),
    lambda x:x['statistics']['global_IUT'].update(global_p_value=0.),
])
def test_partial_misleading_or_resampled_statistics_rejected(matrix,change):
    R=module();x=copy.deepcopy(matrix);change(x)
    with pytest.raises((ValueError,KeyError)):R._layout(x,synthetic_fixture=True)

def test_synthetic_never_crosses_actual_layout_or_public_route(matrix,tmp_path):
    R=module()
    with pytest.raises(ValueError):R._layout(matrix,synthetic_fixture=False)
    p=tmp_path/'fake.json';raw=R.encoded(matrix);p.write_bytes(raw)
    with pytest.raises(ValueError):R.build_layer({'path':str(p),'sha256':R.sha(raw),'size_bytes':len(raw)})
    assert list(tmp_path.iterdir())==[p]

def test_after_execution_source_drift_denied_before_evidence(tmp_path):
    path=tmp_path/PATH.name;path.write_bytes(PATH.read_bytes());R=module(path)
    path.write_bytes(path.read_bytes()+b'\n# drift\n')
    with pytest.raises(ValueError,match='pinned renderer'):R.build_layer({})

def test_whole_typed_report_equality_before_render(tmp_path,monkeypatch,matrix):
    R=module();fake=tmp_path/'analyzer.py'
    # Pure source fixture, not a production analyzer pin or metadata admission.
    fake.write_text("import hashlib\nfrom pathlib import Path\ndef analyze_confirmation_op(request):\n "
        "p=Path(__file__);raw=p.read_bytes();return {'exact':False,'source':{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)},'request_binding':{}}\n")
    monkeypatch.setattr(R,'ANALYZER_SHA',R.sha(fake.read_bytes()))
    actual={'exact':0,'source':{'path':str(fake),'sha256':R.ANALYZER_SHA,'size_bytes':fake.stat().st_size},'request_binding':{}}
    p=tmp_path/'report.json';raw=R.encoded(actual);p.write_bytes(raw)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError,match='typed endpoint/raw replay'):
        R.build_layer({'path':str(p),'sha256':R.sha(raw),'size_bytes':len(raw)})


def test_renderer_known_analyzer_pin():
    R=module();assert R.ANALYZER_SHA==R.sha((PATH.parent/'analyze_confirmation_op_v3.py').read_bytes())


def test_undefined_joint_support_is_never_printed_as_zero(matrix):
    R=module();x=copy.deepcopy(matrix);s=x['statistics'];control=s['controls'][0]
    row=s['comparisons'][control]['per_seed'][0]
    row.update(n_joint=0,joint_instance_ids=[],joint_signed_gap_delta=None)
    row['exact_effects']['joint_gap']=None
    s['comparisons'][control]['mean_joint_signed_gap_delta']=None
    bs=s['graph_bootstrap']['comparisons'][control]
    bs.update(undefined_joint_gap_replicates=50000,mean_joint_gap_delta_ci95=None)
    g=s['global_IUT'];c=g['component_tests'][control]['joint_gap']
    c.update(favorable_by_seed=[False,True,True,True,True],favorable_seeds=4,p_exact={'numerator':'3','denominator':'16'},p_value=3/16)
    g.update(global_p_exact={'numerator':'3','denominator':'16'},global_p_value=3/16,statistical_gate_passed=False)
    x.update(registered_global_IUT_passed=False,scoped_confirmation_established=False)
    text=R._layout(x,synthetic_fixture=True)[R.NAMES[1]]
    assert '0--256' in text and ' & -- & ' in text and 'NOT PASSED' in text


def test_native_size_tables_have_no_resize_and_explicit_feasibility(matrix):
    R=module();text=R._layout(matrix,synthetic_fixture=True)
    for name in R.NAMES[:2]:
        assert 'resizebox' not in text[name]
        assert r'\fontsize{10}{12}' in text[name]
    assert 'Feasible count & $F$ (\\%)' in text[R.NAMES[0]]
    assert r'\begin{tabular}{@{}c@{}}' in text[R.NAMES[1]]


def test_training_reward_reducers_and_evaluation_utility_are_separate(matrix):
    R=module();section=R._layout(matrix,synthetic_fixture=True)[R.NAMES[2]]
    assert r'r=1+P/S' in section and r'r=\kappa-1' in section
    assert 'fixed 3,000-token loss denominator' in section
    assert 'standardized once across the group' in section and 'not head by head' in section
    assert 'differ from the evaluation utility' in section and 'fixed-horizon control' in section
    assert (r'\rho=1' in section) is (matrix['branch']=='teacher')
