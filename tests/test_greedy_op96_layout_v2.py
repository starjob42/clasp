"""Editorial source continuity and actual-caption-order guards; no models."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import pytest

ROOT=Path(__file__).parents[1];D=ROOT/'experiments/rloo_one_shot_20261016/cpu_preparation_reconciliation_v3/greedy_op96_paper_v1'


def load(path):
 raw=path.read_bytes();m=types.ModuleType('synthetic_OP96_layout_guard');m.__file__=str(path);m.__executed_source_sha256__=hashlib.sha256(raw).hexdigest()
 exec(compile(raw,str(path),'exec',dont_inherit=True),m.__dict__);return m


def test_whole_renderer_changes_only_two_float_constants():
 old=(D/'render_greedy_op96_v1.py').read_bytes();new=(D/'render_greedy_op96_v2.py').read_bytes()
 assert hashlib.sha256(old).hexdigest()=='89c108ab000bc32f5aa1d3bd80351357e2193b4ee015297e8eb966d04edc71fc'
 assert old.count(b'\\begin{table}[t]')==2 and new==old.replace(b'\\begin{table}[t]',b'\\begin{table}[H]')


def test_full_admitted_table_and_section_text_changes_only_float_options():
 fixtures=load(ROOT/'tests/test_greedy_op96_paper_v1.py');old=load(D/'render_greedy_op96_v1.py');new=load(D/'render_greedy_op96_v2.py')
 report=fixtures.fixture(old);a=old.layout(report,synthetic_fixture=True);b=new.layout(report,synthetic_fixture=True)
 assert set(a)==set(b) and all(b[name]==text.replace(r'\begin{table}[t]',r'\begin{table}[H]') for name,text in a.items())
 assert a['greedy_op96_section.tex']==b['greedy_op96_section.tex']


def test_existing_main_block_gets_only_one_leading_clearpage():
 m=load(D/'pub_greedy_op96_layout_v2.py');old=m.BEGIN+'\n'+r'\input{generated/greedy_op96_section}'+'\n'+r'\clearpage'+'\n'+m.END+'\n'
 source='OLDRESULTS\n'+old+r'\input{sections/discussion}'+'\n'+r'\bibliographystyle{plainnat}'
 changed=m.patch_main(source.encode());assert changed==source.replace(old,m.BEGIN+'\n'+r'\clearpage'+'\n'+old[len(m.BEGIN)+1:]).encode()
 with pytest.raises(ValueError):m.patch_main(changed)


@pytest.mark.parametrize('bad',['no_existing_layer','duplicate_existing_layer','changed_existing_block'])
def test_unknown_existing_marker_scope_cannot_be_rewritten(bad):
 m=load(D/'pub_greedy_op96_layout_v2.py');old=m.BEGIN+'\n'+r'\input{generated/greedy_op96_section}'+'\n'+r'\clearpage'+'\n'+m.END+'\n'
 text={'no_existing_layer':'','duplicate_existing_layer':old+old,'changed_existing_block':old.replace('greedy_op96_section','something_else')}[bad]
 with pytest.raises(ValueError):m.patch_main((text+r'\input{sections/discussion}'+r'\bibliographystyle{plainnat}').encode())


@pytest.mark.parametrize('rows,passes',[
 ([{'number':10,'page':13,'block':2,'x0':50.,'y0':400.},{'number':11,'page':14,'block':3,'x0':50.,'y0':100.},{'number':12,'page':14,'block':5,'x0':50.,'y0':500.}],True),
 ([{'number':10,'page':13,'block':2,'x0':50.,'y0':400.},{'number':11,'page':13,'block':3,'x0':50.,'y0':100.},{'number':12,'page':14,'block':5,'x0':50.,'y0':500.}],False),
 ([{'number':10,'page':13,'block':2,'x0':50.,'y0':400.},{'number':12,'page':14,'block':3,'x0':50.,'y0':100.},{'number':11,'page':14,'block':5,'x0':50.,'y0':500.}],False),
 ([],False)])
def test_actual_page_and_caption_coordinates_determine_order(tmp_path,monkeypatch,rows,passes):
 m=load(D/'pub_greedy_op96_layout_v2.py');p=load(D/'render_greedy_op96_v1.py').utilities()
 monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout=json.dumps(rows).encode()))
 if passes:assert [r['number'] for r in m.caption_order(tmp_path,p)]==[10,11,12]
 else:
  with pytest.raises(ValueError):m.caption_order(tmp_path,p)


def test_default_publisher_is_inert(tmp_path):
 r=subprocess.run([sys.executable,'-I','-B',str(D/'pub_greedy_op96_layout_v2.py')],cwd=tmp_path,capture_output=True,text=True,check=True)
 assert json.loads(r.stdout)['paper_writes']==0 and not list(tmp_path.iterdir())
