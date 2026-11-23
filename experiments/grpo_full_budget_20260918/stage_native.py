"""Stage an exclusive versioned package and execute its CPU-only preflight."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

SSH = ['ssh', '-T', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20', '-o', 'AddKeysToAgent=no',
       '-o', 'UpdateHostKeys=no', '-o', 'HostName=jean-zay3.idris.fr', '-o', 'HostKeyAlias=jean-zay.idris.fr', 'jean-zay']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, required=True)
    args = parser.parse_args()
    local = Path(__file__).resolve().parent
    package = json.loads((local / f'package_v{args.version}.json').read_text())
    archive = Path(package['archive']['path']).read_bytes()
    assert hashlib.sha256(archive).hexdigest() == package['archive']['sha256']
    draft = Path(package['draft_declaration']['path']).read_bytes()
    assert hashlib.sha256(draft).hexdigest() == package['draft_declaration']['sha256']
    # Only static Python source and base64/JSON literals enter the remote command.
    config = {'package': package, 'version': args.version}
    remote = '''import base64,hashlib,io,json,os,pathlib,subprocess,sys,tarfile
cfg=json.loads(CONFIG_LITERAL)
p=cfg['package']; root=pathlib.Path(p['source']['root']).parent; root.mkdir(parents=True,exist_ok=True)
source=pathlib.Path(p['source']['root']); assert not source.exists(), 'Source already exists; never overwrite a frozen version'
wire=sys.stdin.buffer.read();payload=wire[:p['archive']['size_bytes']];draft=wire[p['archive']['size_bytes']:]
assert hashlib.sha256(payload).hexdigest()==p['archive']['sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tf:
 for member in tf.getmembers():
  relative=pathlib.PurePosixPath(member.name)
  assert relative.parts[0]==source.name and not relative.is_absolute() and '..' not in relative.parts
  assert member.isdir() or member.isfile(), 'Links/devices not admitted'
  target=root.joinpath(*relative.parts)
  if member.isdir(): target.mkdir(parents=True,exist_ok=True)
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f: f.write(tf.extractfile(member).read())
   target.chmod(0o440)
manifest=source/'SOURCE_MANIFEST.sha256'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==p['source']['manifest_sha256']
for line in manifest.read_text().splitlines():
 sha,name=line.split('  ',1); assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
assert hashlib.sha256(draft).hexdigest()==p['draft_declaration']['sha256']
decl=pathlib.Path(p['draft_declaration']['native_path'])
with decl.open('xb') as f:f.write(draft)
decl.chmod(0o440)
d=json.loads(draft); env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
output=root/('cpu_preflight_v'+str(cfg['version']))
print(json.dumps({'status':'staged_exclusive_source','source':str(source),'manifest_sha256':p['source']['manifest_sha256']}),flush=True)
result=subprocess.run([d['runtime_python'],'-B',str(source/'experiments/grpo_full_budget_20260918/preflight.py'),'--declaration',str(decl),'--declaration-sha',p['draft_declaration']['sha256'],'--output',str(output)],cwd=source,env=env)
print(json.dumps({'status':'native_cpu_preflight_exit','returncode':result.returncode,'output':str(output)}),flush=True)
raise SystemExit(result.returncode)
'''.replace('CONFIG_LITERAL', repr(json.dumps(config)))
    result = subprocess.run(SSH + ['python3 -c ' + shlex.quote(remote)], input=archive + draft)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
