"""New layer guards/AST continuity; all staged data are explicit fixtures."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types

import pytest

ROOT=Path(__file__).parents[1]
EXP=ROOT/'experiments/rloo_one_shot_20261016'
SOURCE=EXP/'verify_confirmation_artifacts_v3.py'
OLD=ROOT/'paper/rloo9_20261008/verify_artifacts.py'


def module(paper):
    # Only new stdlib admission nodes execute. Existing NumPy/PDF scientific
    # imports are exercised by the separately recorded real synthetic build.
    tree=ast.parse(SOURCE.read_bytes());nodes=[]
    names={'v3_bytes','v3_binding','v3_input_block','check_one_shot_confirmation_v3'}
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('V3_') for t in n.targets):nodes.append(n)
        if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
    m=types.SimpleNamespace();ns={'Path':Path,'os':os,'hashlib':hashlib,'subprocess':subprocess,'tempfile':tempfile,'json':json,
        'HERE':paper,'ROOT':paper.parents[1],'__file__':str(paper/SOURCE.name),
        '__executed_source_sha256__':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    m.__dict__.update(ns);return m

@pytest.fixture
def layer(tmp_path):
    paper=tmp_path/'paper/manuscript';paper.mkdir(parents=True);(paper/'generated').mkdir()
    (paper/SOURCE.name).write_bytes(SOURCE.read_bytes());(paper/OLD.name).write_bytes(OLD.read_bytes())
    m=module(paper)
    def b(p):
        raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
    (paper/'main.tex').write_text('% BEGIN REGISTERED CONFIRMATION RESULTS\n% END REGISTERED CONFIRMATION RESULTS\n'+m.v3_input_block()+'\\input{sections/discussion}\n\\bibliographystyle{plainnat}\n')
    own=b(paper/SOURCE.name);prefix=str((paper/SOURCE.name).relative_to(m.ROOT))
    entry={'diagnostic_only':True,'status':'synthetic_layout_only','new_tables':2,'new_sources':['generated/'+n for n in m.V3_NAMES],
        'superiority_established':False,'publication_eligible':False,'artifact_verifier_binding':own}
    manifest={'confirmation_OP_layers_v3':[entry],'sources':{prefix:own},'outputs':{}}
    for name in m.V3_NAMES:
        p=paper/'generated'/name;p.write_text('SYNTHETIC LAYOUT FIXTURE; NO MODEL RESULTS. '+name)
        manifest['outputs']['generated/'+name]=b(p)
    return m,paper,manifest,b


def test_all_original_functions_and_import_nodes_preserved_except_registered_layer_guard():
    old=ast.parse(OLD.read_bytes());new=ast.parse(SOURCE.read_bytes())
    oldf={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newf={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert all(newf[name]==node for name,node in oldf.items() if name!='check_registered_confirmation')
    oldi=[ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    newi=[ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all(n in newi for n in oldi)
    assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='3f48a0ad6ed4d94b3fa16bfb0643716015cf6ef751a213f9eee58a00dd65e24c'


def test_explicit_synthetic_stage_is_unpublishable(layer):
    m,p,manifest,b=layer
    result=m.check_one_shot_confirmation_v3(manifest)
    assert result['diagnostic_only'] is True and result['scoped_confirmation_established'] is False
    assert result['status']=='synthetic_layout_only_not_publishable'

@pytest.mark.parametrize('mutation',['extra','missing','unmarked','marker','placement','unbound','original','copied_verifier'])
def test_new_scope_cannot_relax_old_unmanaged_artifact_guard(layer,mutation):
    m,p,manifest,b=layer
    if mutation=='extra':(p/'generated/confirmation_op_extra.tex').write_text('unexpected')
    elif mutation=='missing':(p/'generated'/m.V3_NAMES[0]).unlink()
    elif mutation=='unmarked':
        q=p/'generated'/m.V3_NAMES[0];q.write_text('unmarked');manifest['outputs']['generated/'+q.name]=b(q)
    elif mutation=='marker':(p/'main.tex').write_text((p/'main.tex').read_text()+m.V3_END)
    elif mutation=='placement':(p/'main.tex').write_text('\\input{sections/discussion}'+m.v3_input_block()+'\\bibliographystyle{plainnat}')
    elif mutation=='unbound':manifest['sources']={}
    elif mutation=='original':(p/OLD.name).write_bytes(b'drift')
    else:manifest['confirmation_OP_layers_v3'][0]['artifact_verifier_binding']['sha256']='a'*64
    with pytest.raises((AssertionError,KeyError)):m.check_one_shot_confirmation_v3(manifest)


def test_absent_layer_does_not_allow_stale_v3_files(layer):
    m,p,manifest,b=layer;manifest['confirmation_OP_layers_v3']=[]
    with pytest.raises(AssertionError):m.check_one_shot_confirmation_v3(manifest)


def test_actual_route_cannot_promote_synthetic_text(layer):
    m,p,manifest,b=layer;manifest['confirmation_OP_layers_v3'][0].update(diagnostic_only=False,
        protocol='one-shot-OP-confirmation-paper-layer-v3',status='complete_admitted_confirmation_paper_layer')
    with pytest.raises(AssertionError,match='Synthetic'):m.check_one_shot_confirmation_v3(manifest)


def test_qualified_CPU_subprocess_is_exact_buffer_and_cuda_hidden(layer,monkeypatch):
    m,p,manifest,b=layer;entry=manifest['confirmation_OP_layers_v3'][0]
    entry.update(diagnostic_only=False,protocol='one-shot-OP-confirmation-paper-layer-v3',status='complete_admitted_confirmation_paper_layer',
        scoped_confirmation_established=False,registered_global_IUT_passed=False)
    entry['renderer_binding']=b(EXP/'render_confirmation_op_paper_v3.py')
    q=p/'analysis.json';q.write_text('PURE MOCK; NO ACTUAL REPORT');entry['analysis_binding']=b(q)
    for name in m.V3_NAMES:
        q=p/'generated'/name;q.write_text('PURE MOCK SOURCE '+name);manifest['outputs']['generated/'+name]=b(q)
    seen=[]
    def run(argv,**kw):
        seen.append(argv);assert argv[:4]==[str(Path(m.V3_REPLAY_PYTHON).resolve()),'-I','-B','-c']
        assert argv[6]==m.V3_RENDERER_SHA and kw['env']['CUDA_VISIBLE_DEVICES']=='' and kw['timeout']==3600
        out=Path(argv[-1]);out.mkdir()
        for name in m.V3_NAMES:(out/name).write_bytes((p/'generated'/name).read_bytes())
        rendered={k:entry[k] for k in ('protocol','status','analysis_binding','scoped_confirmation_established','registered_global_IUT_passed')}
        rendered.update(outputs=[{'name':n} for n in m.V3_NAMES],source_bindings=[])
        (out/'render_manifest.json').write_text(json.dumps(rendered))
        return types.SimpleNamespace(returncode=0,stderr=b'')
    monkeypatch.setattr(subprocess,'run',run)
    result=m.check_one_shot_confirmation_v3(manifest)
    assert len(seen)==1 and result['diagnostic_only'] is False
    assert result['status']=='full_V3_confirmation_independent_CPU_replay'
    assert result['scoped_confirmation_established'] is False


def test_old_executed_buffer_cannot_claim_rebound_verifier(layer):
    m,p,manifest,b=layer;q=p/SOURCE.name
    q.write_bytes(q.read_bytes()+b'\n# changed after compile\n')
    row=b(q);manifest['sources'][str(q.relative_to(m.ROOT))]=row
    manifest['confirmation_OP_layers_v3'][0]['artifact_verifier_binding']=row
    with pytest.raises(AssertionError,match='executed V3 verifier'):m.check_one_shot_confirmation_v3(manifest)
