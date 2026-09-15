"""Validate a complete original-control bank without issuing a quality certificate."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_control import digest


def validate(directory, source):
    directory, source = Path(directory), Path(source)
    plan = json.loads((source / 'plan.json').read_text())
    assert all(digest(source / name) == sha for name, sha in plan['sources'].items())
    protocol = json.loads((source / 'protocol.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    assert report['status'] == 'success' and report['quality_certified'] is False
    assert report['checkpoint_unchanged'] and report['attention'] == 'cudnn'
    assert report['plan_sha256'] == digest(source / 'plan.json')
    numeric = report['numeric_attention']
    assert numeric['backend'] == 'cudnn' and numeric['dtype'] == 'bfloat16'
    assert numeric['transformation'] == 'NUMERIC' and numeric['cudnn_version'] == 91501
    assert numeric['eligible_python_calls'] == 128 * 2 * 28
    assert numeric['fallback_python_calls'] == 128 * 2 * 28
    config = report['configuration']
    assert config in protocol['configurations'] and config['id'] in plan['configurations']
    requests = [r for r in protocol['requests'] if r['configuration_id'] == config['id']]
    assert len(requests) == len(report['requests']) == 128
    assets = report['external_runtime_assets']
    assert len(assets) == 1 and assets[0]['bytes'] == 2818839170
    assert assets[0]['sha256'] == '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
    assert report['execution_policy']['precision'] == 'native'
    for index, (row, request) in enumerate(zip(report['requests'], requests, strict=True)):
        assert row['index'] == index and row['json'] == f'{index:04d}.json' and row['npz'] == f'{index:04d}.npz'
        meta, archive = directory / row['json'], directory / row['npz']
        assert digest(meta) == row['json_sha256'] and digest(archive) == row['npz_sha256']
        record = json.loads(meta.read_text())
        assert record['request'] == request and record['arrays_sha256'] == row['npz_sha256']
        trace = record['trace']
        assert trace['observers_restored']
        assert (trace['sampler_calls'], trace['preparation_calls']) == (1, 1)
        assert trace['sampler_denoiser_callbacks'] == config['steps']
        assert trace['model_velocity_branch_calls'] == config['steps'] * (1 if config['guidance'] == 1 else 2)
        assert trace['callback_timesteps'] == ([999., 749., 499., 249.] if config['kind'] == 'unipc' else [1000. * t for t in config['times'][:-1]])
        assert trace['generation_arguments'] == dict(seed=[request['seed']], guidance=config['guidance'],
            num_steps=config['steps'], shift=config['shift'], guidance_interval=None,
            normalize_cfg=False, skip_text_tokens_for_cfg=False)
        assert trace['effective_sampler_arguments'] == dict(num_steps=config['steps'],
            shift=0. if config['kind'] == 'fixed' else config['shift'], seed=[request['seed']])
        if config['action_padding'] == 'zero':
            assert trace['action_padding_intervention']['hooks_restored']
            assert trace['conditioning_observation'] == 'sampler_arguments'
        else:
            assert trace['action_padding_intervention'] is None
            assert trace['conditioning_observation'] == 'native_preparation_callback_context'
        with np.load(archive, allow_pickle=False) as data:
            actions = ('action', 'model_action', 'measured_action', 'model_measured_action')
            fields = ('endpoint', 'reference', 'preserve_mask', 'initial_noise')
            assert set(data.files) == set(actions + fields)
            assert all(data[k].dtype == np.float32 and np.isfinite(data[k]).all() for k in data.files)
            assert all(data[k].shape == (32, 8) for k in actions)
            endpoint = data['endpoint']
            assert endpoint.ndim == 2 and endpoint.shape[0] == 1
            assert all(data[k].shape == endpoint.shape for k in fields)
            assert np.all((data['preserve_mask'] == 0) | (data['preserve_mask'] == 1))
            offset = trace['action_offset']
            assert offset == endpoint.shape[1] - 33 * 64 and trace['action_shape'] == [33, 64]
            action = endpoint[:, offset:].reshape(33, 64)
            assert np.array_equal(action[1:, :8], data['model_action'])
            if config['action_padding'] == 'zero':
                assert np.all(action[:, 8:] == 0)
    return dict(status='success', capture_validation_complete=True, quality_certified=False,
                configuration=config['id'], requests=128, report_sha256=digest(directory / 'report.json'),
                validator_sha256=digest(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = validate(args.directory, args.source)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
