"""Budget, continuation and allocation invariants; no Slurm calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("common", ROOT / "common.py")
common = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = common
spec.loader.exec_module(common)
spec = importlib.util.spec_from_file_location("full_budget_campaign_tested", ROOT / "campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_all_methods_receive_identical_budget_and_initial_diagnostics():
    jobs = common.graph()
    assert len(jobs) == 428
    assert sum(j['gpus'] * j['walltime_seconds'] / 3600 for j in jobs.values()) == 4376
    for task in common.TASKS:
        expected = common.boundaries(task)
        for method in common.METHODS:
            selected = [j for j in jobs.values() if j['task'] == task and j['method'] == method and j['kind'] == 'train']
            assert [j['step'] for j in selected] == expected
            assert sum(j['step'] - j['prior_step'] for j in selected) == common.HORIZONS[task]
            assert selected[0]['depends'] == [common.job_key(task, m, 'diagnostic', 10) for m in common.METHODS]


def test_every_edge_exists_and_graph_is_acyclic():
    jobs = common.graph()
    remaining = set(jobs)
    done = set()
    while remaining:
        ready = {name for name in remaining if set(jobs[name]['depends']) <= done}
        assert ready
        done.update(ready)
        remaining -= ready


def test_mis_crosses_epoch_without_resuming_after_epoch_boundary():
    for method in common.METHODS:
        jobs = common.graph()
        assert common.job_key('mis', method, 'train', 750) not in jobs
        last = jobs[common.job_key('mis', method, 'train', 800)]
        assert last['prior_step'] == 700
        assert last['walltime_seconds'] == 4 * 3600


@pytest.mark.parametrize('task', common.TASKS)
@pytest.mark.parametrize('milestone', [100, 400])
def test_evaluation_blocks_checkpoint_rotation(task, milestone):
    jobs = common.graph()
    for method in common.METHODS:
        next_job = jobs[common.job_key(task, method, 'train', milestone + 50)]
        assert common.job_key(task, method, 'eval', milestone) in next_job['depends']


def allocation():
    binding = {'path': '/tmp/declaration.json', 'sha256': 'a' * 64, 'size_bytes': 1}
    name, job = next(iter(common.graph().items()))
    fields = {'Account': 'vas@a100', 'Partition': 'gpu_p5', 'NumNodes': '1', 'Requeue': '0',
              'Comment': campaign.comment(binding, name), 'JobName': 'fb-' + name,
              'TimeLimit': '01:00:00', 'ReqTRES': 'cpu=40,mem=128G,node=1,billing=4,gres/gpu=4',
              'QOS': 'qos_gpu_a100-dev', 'NumCPUs': '80'}
    return fields, binding, name, job


def test_allocation_accepts_smt_allocation_but_checks_requested_cpu():
    campaign.allocation_checks(*allocation())


@pytest.mark.parametrize('field,value', [('Account', 'wrong'), ('Requeue', '1'),
    ('Comment', 'wrong'), ('NumNodes', '2'), ('TimeLimit', '02:00:00'),
    ('ReqTRES', 'cpu=40,gres/gpu=8'), ('ReqTRES', 'cpu=80,gres/gpu=4'),
    ('QOS', 'qos_gpu_a100-t3')])
def test_allocation_refuses_scope_drift(field, value):
    fields, binding, name, job = allocation()
    fields[field] = value
    with pytest.raises(ValueError):
        campaign.allocation_checks(fields, binding, name, job)


def test_write_once_and_seal_detect_changes(tmp_path):
    path = tmp_path / 'receipt.json'
    common.write_once(path, common.signed({'step': 10}))
    assert common.read(path, sealed=True)['step'] == 10
    with pytest.raises(FileExistsError):
        common.write_once(path, {'step': 10})
    path.chmod(0o640)
    path.write_text(json.dumps({'step': 11, 'receipt_sha256': '0' * 64}))
    with pytest.raises(ValueError):
        common.read(path, sealed=True)


def test_uncertain_submission_never_retries(tmp_path, monkeypatch):
    d = {'root': str(tmp_path), 'jobs': {'x': {}}}
    value = campaign.Campaign(d, {'path': '/tmp/d', 'sha256': 'a' * 64, 'size_bytes': 1})
    value.state['jobs']['x']['status'] = 'submitting'
    monkeypatch.setattr(campaign, 'query_jobs', lambda: {})
    with pytest.raises(RuntimeError, match='Uncertain submission'):
        value.tick(True)


def test_scheduler_failure_stops_new_submissions(tmp_path, monkeypatch):
    d = {'root': str(tmp_path), 'jobs': {'x': {}}}
    value = campaign.Campaign(d, {'path': '/tmp/d', 'sha256': 'a' * 64, 'size_bytes': 1})
    value.state['jobs']['x'].update(status='submitted', job_id='123')
    monkeypatch.setattr(campaign, 'query_jobs', lambda: {})
    monkeypatch.setattr(campaign, 'accounting', lambda _: {'state': 'TIMEOUT', 'exit_code': '0:1'})
    assert value.tick(True) is False
    assert value.state['status'] == 'stopped_for_diagnosis'


def test_no_dependency_gate_bypassed_on_dry_observation(tmp_path, monkeypatch):
    jobs = common.graph()
    d = {'root': str(tmp_path), 'jobs': jobs}
    value = campaign.Campaign(d, {'path': '/tmp/d', 'sha256': 'a' * 64, 'size_bytes': 1})
    monkeypatch.setattr(campaign, 'query_jobs', lambda: {})
    monkeypatch.setattr(value, 'initialization', lambda task: task != 'tsp')
    assert value.tick(False)
    assert len(value.state['eligible']) == 16
    assert all(jobs[name]['kind'] == 'diagnostic' and jobs[name]['step'] == 5 for name in value.state['eligible'])
