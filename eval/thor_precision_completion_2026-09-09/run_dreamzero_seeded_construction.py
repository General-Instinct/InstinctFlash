"""Run a single explicitly named Thor diagnostic with an exclusive GPU lock."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('precision', choices=('native', 'fp8'))
parser.add_argument('run_id')
parser.add_argument('--preflight-only', action='store_true')
args = parser.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run_id):
    raise ValueError('Invalid run id')
manifest = json.loads((root/'bundle.json').read_text())
for relative, expected in manifest['files'].items():
    path = root/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Changed bundle file: {relative}')
actual = {str(p.relative_to(root/'source')) for p in (root/'source').rglob('*')
          if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
expected = {p[len('source/'):] for p in manifest['files'] if p.startswith('source/')}
if actual != expected:
    raise ValueError('Source inventory changed')
if args.preflight_only:
    print(json.dumps({'preflight': 'pass', 'bundle_files': len(manifest['files']),
                      'scope': 'Files only; no model imported or GPU execution'}))
    sys.exit(0)
python = '/home/guanming/thorcol/dz_env/bin/python'
env = dict(os.environ)
env.update(PYTHONPATH=f'{root}/source:{root}/source/examples/dreamzero:/home/guanming/thorcol/dreamzero',
           PATH='/home/guanming/thorcol/dz_env/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
           OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
           TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
           TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas')
with open('/tmp/thor_gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root/f'{args.precision}-{args.run_id}.log').open('x') as log:
        subprocess.run([python, str(root/'probe_dreamzero_seeded_construction.py'),
                        args.precision, args.run_id], env=env, cwd=root,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
