"""Synthetic table/metadata ordering diagnostics; no policy or model calls."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import pytest

ROOT=Path(__file__).parents[1]
SOURCE=ROOT/'experiments/rloo_one_shot_20261016/cpu_preparation_reconciliation_v3/greedy_op96_paper_v1/render_greedy_op96_v1.py'


def load():
 raw=SOURCE.read_bytes();m=types.ModuleType('synthetic_greedy_OP96_renderer');m.__file__=str(SOURCE);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 exec(compile(raw,str(SOURCE),'exec',dont_inherit=True),m.__dict__);return m


def fixture(m):
 summaries={arm:{'N':96,'n_feasible':80+i,'feasibility_percent':100*(80+i)/96,'own_feasible_signed_gap_percent':-3.+i/10.,'all_instance_u10_percent':60.+i} for i,arm in enumerate(m.ORDER)}
 comparisons={}
 for i,arm in enumerate(m.CONTROLS):
  delta=100*(summaries['one_shot_rloo']['n_feasible']-summaries[arm]['n_feasible'])/96
  comparisons[arm]={'N':96,'n_joint':75+i,'bootstrap_replicates':50000,'bootstrap_seed':2026101602,'interval':'pointwise95_percentile',
   'uncertainty_scope':'fixed_policies_saved_pools_and_cohort;not_training_seed_or_familywise',
   'feasibility_delta_pp':{'estimate':delta,'ci95':[-1.,8.]},'joint_signed_gap_delta_pp':{'estimate':.25+i,'ci95':[-.5,4.]},'u10_delta_pp':{'estimate':1.,'ci95':[-2.,4.]}}
 return {'protocol':'independent-one-shot-development-replay-v1','status':'complete_task_raw_replay','task':'op','quality_analyzed':True,
  'superiority_established':False,'development_screen_passed':False,'summaries':summaries,'rloo_minus_control_comparisons':comparisons}


def test_default_has_no_outputs(tmp_path):
 r=subprocess.run([sys.executable,'-I','-B',str(SOURCE)],cwd=tmp_path,capture_output=True,text=True,check=True)
 assert json.loads(r.stdout)['paper_writes']==0 and not list(tmp_path.iterdir())


def test_synthetic_full_layout_has_support_and_claim_boundaries():
 m=load();texts=m.layout(fixture(m),synthetic_fixture=True);alltext='\n'.join(texts.values())
 assert set(texts)==set(m.NAMES) and all('SYNTHETIC LAYOUT FIXTURE' in t for t in texts.values())
 assert all(label in alltext for label in ('Joint $n$','Own gap','one actual greedy','reward matched','not certified optima','screen fails'))
 assert all('resizebox' not in t for t in texts.values()) and sum('\\begin{table}' in t for t in texts.values())==2
 assert all('\\fontsize{10}{12}' in texts[name] for name in m.NAMES[:2])
 assert 'confirmation_' not in alltext and 'teacher_op_development' not in alltext


@pytest.mark.parametrize('change',[lambda r:r.update(development_screen_passed=True),lambda r:r.update(superiority_established=True),
 lambda r:r['summaries']['one_shot_rloo'].update(N=True),lambda r:r['summaries']['foarl'].update(n_feasible=True),
 lambda r:r['summaries']['foarl'].update(feasibility_percent=99.),lambda r:r['summaries']['foarl'].update(own_feasible_signed_gap_percent=float('nan')),
 lambda r:r['rloo_minus_control_comparisons']['foarl'].update(n_joint=96),lambda r:r['rloo_minus_control_comparisons']['foarl'].update(bootstrap_replicates=5000),
 lambda r:r['rloo_minus_control_comparisons']['foarl']['feasibility_delta_pp'].update(estimate=99.),
 lambda r:r['rloo_minus_control_comparisons']['foarl']['joint_signed_gap_delta_pp'].update(ci95=[2.,-1.])])
def test_corrupt_scientific_display_or_claim_does_not_render(change):
 m=load();report=fixture(m);change(report)
 with pytest.raises(ValueError):m.layout(report)


def test_incomplete_metadata_stops_before_any_report_read(tmp_path,monkeypatch):
 m=load();u=m.utilities();request={k:{'path':'/explicitly_synthetic/'+k,'sha256':v,'size_bytes':1} for k,v in m.PINS.items()}
 request.update(protocol=m.REQUEST_PROTOCOL,analysis_entry_source={'sha256':m.KNOWN_SHIM_SHA},fresh_replay_report={'path':'/NEVER/read/raw-quality'},
 operational_receipt={'sha256':m.KNOWN_REPLAY_RECEIPT_SHA},runtime_context={},runtime_context_parent={})
 request['receipt_sha256']=u.sha(u.encoded(request));p=tmp_path/'synthetic_request.json';p.write_bytes(u.encoded(request));row=u.binding(p)
 reads=[];original=u.read_bound
 def read(bound):
  reads.append(bound)
  if bound!=row:pytest.fail('Incomplete task attempted downstream report/source read')
  return original(bound)
 u.read_bound=read;u.module=lambda *a:types.SimpleNamespace(admit_complete_task=lambda *a:{'status':'incomplete_task_no_quality_analysis'})
 monkeypatch.setattr(m,'utilities',lambda:u)
 with pytest.raises(ValueError,match='All five completed'):m.build_layer(row)
 assert reads==[row]


def test_synthetic_marker_is_rejected_before_metadata_module(tmp_path,monkeypatch):
 m=load();u=m.utilities();request={'synthetic_fixture':True};request['receipt_sha256']=u.sha(u.encoded(request));p=tmp_path/'request.json';p.write_bytes(u.encoded(request))
 monkeypatch.setattr(m,'utilities',lambda:u);u.module=lambda *a:pytest.fail('Fixture may not reach metadata admission')
 with pytest.raises(ValueError,match='Synthetic'):m.build_layer(u.binding(p))


def test_actual_numeric_layer_requires_its_executed_buffer_identity(tmp_path):
 m=load();m.__executed_source_sha256__='0'*64
 with pytest.raises(ValueError,match='Externally pinned'):m.build_layer({})
