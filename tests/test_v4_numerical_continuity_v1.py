"""Reject hidden numerical/global/CLI/import changes after actual GPU evidence."""
import ast
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/rloo_sft_anchor_20261013'
spec=importlib.util.spec_from_file_location('v4_continuity',EXP/'verify_v4_numerical_continuity_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OLD=EXP/'validation/best8_cuda_v4_v1/executed_source_0608/files'/m.ENTRY
FINAL=ROOT/m.ENTRY


def test_whole_ast_keeps_numeric_functions_global_context_classes_and_existing_cli_defaults():
    old,_=m.normalized(OLD.read_bytes(),final=False);new,_=m.normalized(FINAL.read_bytes(),final=True)
    assert m.dump(old)==m.dump(new)


@pytest.mark.parametrize('kind',['decoder','dtype','groups','CLI_default','import','source_pin'])
def test_unreviewed_numeric_global_or_cli_changes_break_complete_ast_continuity(kind):
    raw=FINAL.read_text()
    if kind=='decoder':raw=raw.replace('top_p=settings.get("top_p", 1.)','top_p=settings.get("top_p", .9)')
    elif kind=='dtype':raw=raw.replace('dtype = torch.float32 if contract["scope"] == "system_smoke" else torch.bfloat16',
                                    'dtype = torch.float16 if contract["scope"] == "system_smoke" else torch.bfloat16')
    elif kind=='groups':raw=raw.replace('"RL_responses": steps * 36','"RL_responses": steps * 72')
    elif kind=='CLI_default':raw=raw.replace('choices=("cpu", "cuda"), default="cpu"','choices=("cpu", "cuda"), default="cuda"')
    elif kind=='import':raw='import random\n'+raw
    else:raw=raw.replace(m.EXECUTED_SHA[:4],'ffff') if m.EXECUTED_SHA[:4] in raw else raw.replace('STATISTICAL_SHA = "9c474','STATISTICAL_SHA = "fffff')
    old,_=m.normalized(OLD.read_bytes(),final=False);new,_=m.normalized(raw,final=True)
    assert m.dump(old)!=m.dump(new)
