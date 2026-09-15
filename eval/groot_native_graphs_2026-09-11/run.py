"""Sequential fresh-process Thor ablation under the shared GPU lock."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root
output.mkdir(parents=True, exist_ok=True)
source = root / 'eval/groot_native_graphs_2026-09-11'
lock = open('/tmp/thor_gpu.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
py = '/home/guanming/thorcol/groot_env/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1',
    PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
    GR00T_ROOT='/home/guanming/thorcol/Isaac-GR00T',
    TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', TRITON_PTXAS_BLACKWELL_PATH='/usr/local/cuda/bin/ptxas',
    PYTHONPATH=f'{root}:{root}/examples/groot_n17',
    PATH=f'{Path(py).parent}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin')
for variant in ('baseline_a', 'graph_only', 'fused_graph', 'baseline_b'):
    out = output / (variant + '.json')
    assert not out.exists()
    env['IFL_BENCH_VARIANT'] = variant
    env['IFL_BENCH_PROFILE'] = '1' if variant == 'baseline_a' else '0'
    with out.with_suffix('.log').open('x') as log:
        run = subprocess.run([py, str(source / 'benchmark.py'), 'groot', 'native', str(out),
            '--iterations', '20', '--input-archive', str(root / 'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')],
            env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    report = json.loads(out.read_text()) if out.exists() else {}
    print(dict(variant=variant, exit_code=run.returncode, p50_ms=report.get('p50_ms'), error=report.get('error')), flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)
