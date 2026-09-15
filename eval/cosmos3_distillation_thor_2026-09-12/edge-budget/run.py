"""Run original-weight Edge budget arms in fresh processes under the Thor lock."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--fixture', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
with open('/tmp/thor_gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for steps, guidance, attention in ((1, 1, 'native'), (1, 1, 'cudnn'), (1, 4, 'cudnn')):
        output = a.output.resolve() / f's{steps}-cfg{guidance}-{attention}.json'
        with output.with_suffix('.log').open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('benchmark.py')),
                '--steps', str(steps), '--guidance', str(guidance), '--attention', attention,
                '--fixture', str(a.fixture.resolve()), str(output)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400,
                env=os.environ.copy())
