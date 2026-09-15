"""Serialize the two actual-student latency arms under the shared Thor lock."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--fixture',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
with open('/tmp/thor_gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for arm in ('native','cudnn'):
        output=a.output.resolve()/f'{arm}.json'
        with output.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark_candidate.py')),
                str(a.checkpoint.resolve()),str(output),'--allow-unqualified','--attention',arm,'--fixture',str(a.fixture.resolve())],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400,env=os.environ.copy())
