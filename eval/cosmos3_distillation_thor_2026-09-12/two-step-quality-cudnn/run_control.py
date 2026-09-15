"""Frozen native Thor original controls; raw generation timings are not benchmarks."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['status'] == 'frozen'
    assert all(digest(root / name) == sha for name, sha in plan['sources'].items())
    protocol = json.loads((root / 'protocol.json').read_text())
    config = next(c for c in protocol['configurations'] if c['id'] == args.configuration)
    assert args.configuration in plan['configurations'] and config['weight'] == 'original'
    checkpoint = Path(plan['checkpoint'])
    inventory = json.loads((root / 'inventory.json').read_text())

    def verify_checkpoint():
        for item in inventory['files']:
            path = checkpoint / item['relative_path']
            assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], path
        assert digest(checkpoint / 'instinctflash.json') == plan['declaration_sha256']

    verify_checkpoint()
    data_path = Path(plan['data']['path'])
    assert data_path.stat().st_size == plan['data']['bytes'] and digest(data_path) == plan['data']['sha256']
    requests = [r for r in protocol['requests'] if r['configuration_id'] == args.configuration]
    assert len(requests) == 128
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', quality_certified=False, configuration=config, attention='cudnn',
                  plan_sha256=digest(root / 'plan.json'), scope=plan['scope'], requests=[])
    api = None
    with open('/tmp/thor_gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for key in list(os.environ):
                if key.startswith('IFL_COSMOS3_'):
                    del os.environ[key]
            os.environ.update(IFL_COSMOS3_EXACT_POINTWISE='1', IFL_COSMOS3_LAYER_GRAPHS='1', IFL_COSMOS3_ATTENTION='native')
            import torch
            from instinctflash import Runtime
            from runtime_assets import audit_vae_load
            from capture_control import capture_control
            assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (11, 0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            with audit_vae_load() as assets:
                api = Runtime.from_pretrained(checkpoint, strict=False, precision='native', placement='in_process',
                                              tier_ceiling='numeric')
                api.reset(prompt='initialize original control')
            service = api._backend._impl._service
            import cosmos_framework
            import cosmos_framework.model.generator.mot.attention as mot
            from cosmos3_iwm.conditioning_cache import SOURCE_HASHES
            from cosmos3_iwm.numeric_attention import NumericAttention
            from instinctflash.planners.planner import PassResult, Tier
            vendor = Path(cosmos_framework.__file__).parent
            assert all(digest(vendor/name) == sha for name,sha in SOURCE_HASHES.items())
            assert torch.backends.cudnn.version() == 91501
            owners = [layer.self_attn for layer in service.model.net.language_model.model.layers]
            assert len(owners) == 28 and all(o.dispatch_attention_fn is mot.dispatch_attention for o in owners)
            service._ifl_numeric_attention = NumericAttention(mot, owners)
            api.plan.results.append(PassResult('experimental_original_sde2_cudnn', True, Tier.NUMERIC,
                'Paired historical diagnostic; no task quality certificate'))
            report['external_runtime_assets'] = assets
            with np.load(data_path, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            for index, request in enumerate(requests):
                pids = subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid',
                                                '--format=csv,noheader,nounits'], text=True).splitlines()
                assert not [pid for pid in pids if int(pid) != os.getpid()], 'GPU contention'
                i = request['source_index']
                assert (str(data['episode_id'][i]), int(data['frame_index'][i])) == (request['episode_id'], request['frame_index'])
                observation = dict(image=data['image'][i].copy(), state=np.asarray(data['state'][i], np.float32),
                                   prompt=str(data['prompt'][i]))
                api.reset(prompt=observation['prompt'])
                arrays, fields, trace = capture_control(service, observation, data['measured_action'][i], request['seed'], config)
                path = args.output / f'{index:04d}.npz'
                np.savez_compressed(path, **arrays, **fields)
                record = dict(request=request, arrays_sha256=digest(path), trace=trace)
                meta = path.with_suffix('.json')
                meta.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
                report['requests'].append(dict(index=index, json=meta.name, json_sha256=digest(meta),
                                               npz=path.name, npz_sha256=digest(path)))
                print(json.dumps(dict(configuration=args.configuration, completed=index+1, total=128)), flush=True)
            report['numeric_attention'] = service._ifl_numeric_attention.report()
            assert report['numeric_attention']['eligible_python_calls'] > 0
            assert report['numeric_attention']['eligible_python_calls'] == 128 * 2 * 28
            assert report['numeric_attention']['fallback_python_calls'] == 128 * 2 * 28
            verify_checkpoint()
            report.update(status='success', checkpoint_unchanged=True, torch=torch.__version__,
                          cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                          execution_policy=api.execution_policy, backend_stats=api._backend._impl.backend_stats())
        except BaseException:
            report.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            report['sources'] = {str(Path(m.__file__).resolve()): digest(m.__file__)
                for name, m in list(sys.modules.items()) if name.startswith(('instinctflash', 'instinct_compress',
                    'cosmos3_iwm', 'cosmos_framework', 'runtime_assets', 'capture_control', 'producer_capture'))
                and getattr(m, '__file__', None) and str(m.__file__).endswith('.py') and Path(m.__file__).is_file()}
            if api is not None:
                api.close()
            (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
