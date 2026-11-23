"""Finalize a passed native CPU preflight and optionally start the bounded controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from stage_native import SSH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    local = Path(__file__).resolve().parent
    package = json.loads((local / f'package_v{args.version}.json').read_text())
    cfg = {'package': package, 'version': args.version, 'execute': args.execute}
    code = '''import hashlib,json,os,pathlib,socket,subprocess,sys,time
cfg=json.loads(CONFIG_LITERAL);p=cfg['package'];source=pathlib.Path(p['source']['root']);root=source.parent
sys.path.insert(0,str(source/'experiments/grpo_full_budget_20260918'))
from common import canonical,identity,read,require,verify,write_once
draft=pathlib.Path(p['draft_declaration']['native_path']);require(identity(draft)['sha256']==p['draft_declaration']['sha256'],'Draft changed')
d=read(draft);preflight=root/('cpu_preflight_v'+str(cfg['version']))/'complete.json';passed=read(preflight,sealed=True)
require(passed['status']=='passed' and passed['numerical_declaration_sha256']==hashlib.sha256(canonical(d)).hexdigest(),'Preflight does not cover numerical declaration')
require(passed['source']==d['source'] and len(passed['reports'])==20,'Preflight source/case coverage differs')
for bound in passed['reports'].values():verify(bound)
d['cpu_preflight']=identity(preflight);decl=root/('launch_declaration_v'+str(cfg['version'])+'.json')
if decl.exists():require(read(decl)==d,'Existing final declaration changed')
else:write_once(decl,d)
binding=identity(decl);env=dict(os.environ,PYTHONPATH=str(source/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[d['runtime_python'],'-B',str(source/'experiments/grpo_full_budget_20260918/campaign.py'),'--declaration',str(decl),'--sha',binding['sha256']]
controller=None
if cfg['execute']:
 env.pop('CUDA_VISIBLE_DEVICES',None)  # Slurm supplies each allocation's GPU visibility.
 out=pathlib.Path(d['root'])/'controller.log';out.parent.mkdir(parents=True,exist_ok=True)
 # Only one fresh bootstrap is permitted. A restart requires separate diagnosis.
 boot=pathlib.Path(d['root'])/'controller_bootstrap.json';require(not boot.exists(),'Controller already bootstrapped')
 write_once(boot,{'command':command+['--execute'],'declaration':binding,'started_at_unix':time.time(),'host':socket.gethostname()})
 with out.open('ab',buffering=0) as log:
  process=subprocess.Popen(command+['--execute'],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 controller={'pid':process.pid,'host':socket.gethostname(),'command':command+['--execute'],'log':str(out)}
 write_once(pathlib.Path(d['root'])/'controller_process.json',controller)
else:
 result=subprocess.run(command+['--once'],cwd=source,env=env,capture_output=True,text=True,timeout=300)
 require(result.returncode==0,'Dry admission failed: '+result.stderr[-8000:])
state=read(pathlib.Path(d['root'])/'campaign_status.json',sealed=True)
from campaign import submission_command
preview=[{'name':name,'command':submission_command(d,binding,name,d['jobs'][name])} for name in state.get('eligible',[])[:2]]
reports={name:read(bound['path'],sealed=True) for name,bound in passed['reports'].items()}
print(json.dumps({'status':'bounded_controller_started' if cfg['execute'] else 'passed_native_dry_admission','declaration':binding,'declaration_value':d,'preflight':passed,'preflight_reports':reports,'state':state,'next_commands':preview,'controller':controller}))
'''.replace('CONFIG_LITERAL', repr(json.dumps(cfg)))
    result = subprocess.run(SSH + ['python3 -c ' + shlex.quote(code)], text=True, capture_output=True)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)
    value = json.loads(result.stdout)
    label = 'native_launch' if args.execute else 'native_admission'
    path = local / f'{label}_v{args.version}.json'
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if not args.execute:
        declaration = local / f'launch_declaration_v{args.version}.json'
        with declaration.open('x') as stream:
            # Preserve native serialization so its SHA is reproducible locally.
            json.dump(value['declaration_value'], stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        declaration.chmod(0o440)
    print(json.dumps({'status': value['status'], 'declaration': value['declaration'],
                      'eligible': value['state'].get('eligible'), 'controller': value['controller']}, indent=2))


if __name__ == '__main__':
    main()
