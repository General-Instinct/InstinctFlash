"""One fresh-process Thor Cosmos DROID regression arm (full 32x8 actions)."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from instinctflash import Runtime

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['edge', 'nano'])
    parser.add_argument('arm', choices=['baseline_a', 'current', 'baseline_b'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'eval/native_total_2026-09-10/fixtures/va_eval_obs.npz')
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 10:
        parser.error('At least 10 measured requests are required')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('Output already exists')
    assert torch.cuda.get_device_capability() == (11, 0), 'This protocol targets Thor'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model_id = 'nvidia/Cosmos3-' + ('Edge' if args.family == 'edge' else 'Nano') + '-Policy-DROID'
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(model_id, local_files_only=True))
    with np.load(args.fixture, allow_pickle=True) as archive:
        frames = [np.asarray(Image.open(io.BytesIO(bytes(row[0]))).convert('RGB').resize((640, 540))) for row in archive['jpeg_0'][:12]]
    prompts = ['pick up the object', 'place the object down']
    cases = []
    # Exercise both prompt geometries before steady-state measurement, while
    # retaining their cold actions/latencies and checking them across arms.
    for i in range(6 + args.iterations):
        cases.append({'i': i, 'phase': 'warmup' if i < 6 else 'measured',
                      'frame': i % len(frames), 'prompt': prompts[(i // 3) % 2],
                      'reset': i < 6 or i % 5 == 0, 'state': .01 * (i % 3), 'seed': 707 + i})
    report = {'family': args.family, 'arm': args.arm, 'model_id': model_id,
              'revision': snapshot.name, 'precision': 'native', 'device': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'input_archive_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'reference_constructor_sha256': hashlib.sha256((ROOT / 'eval/native_total_2026-09-10/upstream.py').read_bytes()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': cases, 'calls': [], 'competing_gpu_processes': [], 'ok': False}
    api = None
    actions = []
    def contention(label):
        lines = subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
        others = [int(x) for x in lines if x.strip() and int(x) != os.getpid()]
        if others:
            report['competing_gpu_processes'].append({'case': label, 'pids': others})
        return others
    try:
        if contention('before_load'):
            raise RuntimeError('Thor is occupied')
        api = Runtime.from_pretrained(model_id, revision=snapshot.name, precision='native', tier_ceiling='bitexact')
        if args.arm != 'current':
            sys.path.insert(0, str(ROOT / 'eval/native_total_2026-09-10'))
            from upstream import build
            api = build(args.family, api)
        api.reset(prompt=prompts[0])
        report['default_schedule'] = dict(api._checkpoint.execution.nfe or {})
        report['guidance'] = str(api._checkpoint.execution.guidance)
        for case in cases:
            contention(case['i'])
            if case['reset']:
                api.reset(prompt=case['prompt'])
            seed = case['seed']
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            obs = {'image': frames[case['frame']].copy(), 'state': np.full(8, case['state'], np.float32), 'prompt': case['prompt']}
            torch.cuda.synchronize()
            start = time.perf_counter()
            action = np.asarray(api.predict(obs)['action'])
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - start)
            if action.shape != (32, 8) or not np.isfinite(action).all():
                raise ValueError('Invalid full DROID action chunk')
            actions.append(action.copy())
            row = {'i': case['i'], 'phase': case['phase'], 'ms': elapsed}
            report['calls'].append(row)
            print(json.dumps(row), flush=True)
        loop = api._backend._impl
        report['backend_stats'] = loop.backend_stats()
        stats = report['backend_stats']
        report['effective_schedule'] = {k: stats[k] for k in ('action_chunk_size', 'action_steps', 'guidance', 'conditioning_fps')}
        import triton
        report['runtime_versions'] = {'cuda': torch.version.cuda, 'triton': triton.__version__, 'python': sys.version}
        report['execution_policy'] = api.execution_policy
        report['numeric_environment'] = {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark}
        np.savez_compressed(args.output.with_suffix('.npz'), actions=np.stack(actions))
        report['actions_sha256'] = hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest()
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['ok'] = True
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report['sources'] = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for name, m in list(sys.modules.items()) if name.startswith(('instinctflash', 'cosmos3_iwm', 'cosmos_framework'))
            and getattr(m, '__file__', None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
        if api is not None:
            api.close()
        args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
