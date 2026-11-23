"""Read-only bounded local collection of native status and signed comparisons."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from common import HORIZONS, METHODS, atomic, canonical, identity, read, require, signed
from stage_native import SSH


def verify_payload(raw, binding):
    require(len(raw) == binding['size_bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Collected bytes differ')
    value = json.loads(raw)
    require(value == signed({k: v for k, v in value.items() if k != 'receipt_sha256'}), 'Collected receipt hash differs')
    return value


def fetch(declaration):
    root = declaration['root']
    source = '''import base64,hashlib,json,pathlib
root=pathlib.Path(ROOT_LITERAL)
state_path=root/'campaign_status.json';raw=state_path.read_bytes();state=json.loads(raw)
def bound(p,b):return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
items={'campaign_status.json':{'binding':bound(state_path,raw),'bytes':base64.b64encode(raw).decode()}}
for name,binding in state.get('reports',{}).items():
 p=pathlib.Path(binding['path']);assert p.parent==root/'reports' and p.name==name+'.json'
 b=p.read_bytes();assert bound(p,b)==binding
 items['reports/'+p.name]={'binding':binding,'bytes':base64.b64encode(b).decode()}
process=json.loads((root/'controller_process.json').read_text())
try:command=pathlib.Path('/proc',str(process['pid']),'cmdline').read_bytes().rstrip(b'\\0').decode().split('\\0')
except FileNotFoundError:command=[]
print(json.dumps({'items':items,'controller':{'expected':process,'alive_matching_command':command==process['command']}}))
'''.replace('ROOT_LITERAL', repr(root))
    result = subprocess.run(SSH + ['python3 -'], input=source, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout)


def render(local, d, state, reports, controller):
    counts = {}
    for value in state['jobs'].values():
        counts[value['status']] = counts.get(value['status'], 0) + 1
    lines = ['Native full-budget campaign status: **' + state['status'] + '**.', '',
             'Observed at ' + dt.datetime.now(dt.timezone.utc).isoformat() + '.', '',
             'Allocations: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())) + '.', '',
             'Full-budget superiority remains unestablished until comparable results are available.', '']
    if state['status'] == 'monitoring' and not controller['alive_matching_command']:
        lines += ['The controller process was not observed. Existing allocations may continue, but automatic advancement requires diagnosis.', '']
    active = [(name, rec) for name, rec in state['jobs'].items() if rec['status'] in ('submitted', 'submitting')]
    if active:
        lines += ['| Active allocation | Job ID | Observed state |', '|---|---:|---|']
        lines += [f"| {name} | {rec.get('job_id', 'uncertain')} | {rec.get('observed_state', rec['status'])} |" for name, rec in active]
        lines += ['']
    failures = [(name, rec) for name, rec in state['jobs'].items() if rec['status'].startswith('failed')]
    if failures:
        lines += ['Stopped for diagnosis: ' + ', '.join(name for name, _ in failures) + '.', '']
    if not reports:
        lines += ['No four-method milestone comparison is complete yet. [Existing 100-update results](../rloo_b9_completion_20260918/README.md) remain a separate experiment.', '']
    for key in sorted(reports, key=lambda name: (name.split('_step')[0], int(name.split('_step')[1]))):
        report = reports[key]
        lines += [f"**{report['task'].upper()}, update {report['step']}/{report['horizon']}** — [signed report](reports/{key}.json).", '',
                  '| Method | Feasibility@1 | Feasibility@8 | Signed reference gap, own successes |', '|---|---:|---:|---:|']
        for row in report['summary']:
            gap = row['mean_signed_reference_gap_percent']
            gap_text = 'undefined' if gap is None else f'{gap:.6f}%'
            lines.append(f"| {row['method']} | {row['feasible_at_1_percent']:.1f}% | {row['feasible_at_8_percent']:.1f}% | {gap_text} (n={row['own_success_gap_denominator']}) |")
        lines += ['']
    lines += ['Lower signed gap is better; negative values beat the declared reference. Own-success gaps may use different instance sets. Signed reports contain paired joint-success comparisons and instance-bootstrap intervals. These are one-seed results on inspected DEV100 cohorts.', '',
              '[Experiment and fairness limits](README.md). The collector reads native results and does not submit, retry, cancel or modify remote work.', '']
    target = local / 'STATUS.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines))
    temporary.replace(target)


def terminal_summary(local, reports):
    terminal = {value['task']: value for value in reports.values() if value['step'] == value['horizon']}
    comparisons = {}
    for candidate in ('rloo_legacy', 'rloo_quality'):
        contrasts = []
        for task, report in sorted(terminal.items()):
            rows = {row['method']: row for row in report['summary']}
            for baseline in ('classic_grpo', 'foarl'):
                paired = report['paired_comparisons'][baseline + '_vs_' + candidate]
                gap = paired['joint_signed_gap']['right_minus_left']
                feasibility = rows[candidate]['feasible_at_8_percent'] - rows[baseline]['feasible_at_8_percent']
                contrasts.append({'task': task, 'baseline': baseline,
                    'feasibility_at_8_difference_pp': feasibility,
                    'joint_successes': paired['joint_success_count'],
                    'paired_gap_difference_pp': None if gap is None else gap * 100,
                    'paired_gap_bootstrap_fraction_units': paired['joint_signed_gap']['paired_bootstrap'],
                    'point_estimate_gap_improves_without_lower_feasibility': feasibility >= 0 and gap is not None and gap < 0})
        comparisons[candidate] = {'contrasts': contrasts,
            'all_five_tasks_available': set(terminal) == set(HORIZONS),
            'improves_paired_gap_without_lower_f8_against_both_baselines_on_all_five_tasks':
                len(contrasts) == 10 and all(row['point_estimate_gap_improves_without_lower_feasibility'] for row in contrasts)}
    atomic(local / 'terminal_comparison_status.json', {
        'status': 'complete_development_point_comparisons' if len(terminal) == 5 else 'incomplete',
        'available_tasks': sorted(terminal), 'candidate_comparisons': comparisons,
        'training_seeds': [43], 'confirmation_established': False, 'guaranteed_superiority': False,
        'interpretation': 'Candidate minus baseline; lower signed gap is better. This is a descriptive one-seed development comparison, not a significance or generalization claim.'})


def observe(local, declaration_path):
    d = read(declaration_path)
    declared_binding = identity(declaration_path)
    # The native absolute path differs from the local exact-byte copy.
    binding = {**declared_binding, 'path': str(Path(d['root']).parent / declaration_path.name)}
    fetched = fetch(d)
    items, controller = fetched['items'], fetched['controller']
    state_raw = base64.b64decode(items['campaign_status.json']['bytes'], validate=True)
    state = verify_payload(state_raw, items['campaign_status.json']['binding'])
    require(state['declaration'] == binding and set(state['jobs']) == set(d['jobs']), 'Collected campaign differs')
    reports = {}
    for name, item in items.items():
        raw = base64.b64decode(item['bytes'], validate=True)
        value = verify_payload(raw, item['binding'])
        if name != 'campaign_status.json':
            key = Path(name).stem
            require(state['reports'][key] == item['binding'] and value['campaign_declaration'] == binding,
                    'Collected report belongs to another campaign')
            require(value['status'] == 'complete_four_methods_within_task_milestone'
                    and value['horizon'] == HORIZONS[value['task']]
                    and value['step'] in (100, 400, value['horizon'])
                    and {row['method'] for row in value['summary']} == set(METHODS), 'Incomplete collected comparison')
            require(all(row['instances'] == 100 and row['candidate_count'] == 800 for row in value['summary']), 'Wrong comparison denominators')
            reports[key] = value
            target = local / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(target.read_bytes() == raw, 'Archived report changed')
            else:
                target.write_bytes(raw)
                target.chmod(0o440)
    atomic(local / 'current_status.json', state)
    render(local, d, state, reports, controller)
    terminal_summary(local, reports)
    atomic(local / 'collector_status.json', {'status': 'observed', 'observed_at_unix': time.time(),
           'campaign_status': state['status'], 'reports': list(reports), 'controller': controller,
           'remote_scheduler_mutations': False})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    local = Path(__file__).resolve().parent
    deadline = time.time() + 61 * 86400
    with (local / '.collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < deadline:
            try:
                state = observe(local, args.declaration.absolute())
                print(json.dumps({'status': state['status'], 'reports': len(state.get('reports', {})), 'observed_at_unix': time.time()}), flush=True)
                if state['status'] in ('complete', 'stopped_for_diagnosis'):
                    return
            except Exception as exc:
                atomic(local / 'collector_status.json', {'status': 'observation_error', 'observed_at_unix': time.time(),
                       'error': f'{type(exc).__name__}: {exc}', 'remote_scheduler_mutations': False})
                if not args.watch:
                    raise
            if not args.watch:
                return
            time.sleep(300)


if __name__ == '__main__':
    main()
