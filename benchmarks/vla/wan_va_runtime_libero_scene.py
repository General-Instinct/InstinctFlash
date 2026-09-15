"""One frozen LIBERO scene using VA Runtime's observed-history protocol."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from . import wan_va_libero_driver as driver
from .remote_policy import RemotePolicy
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic
from .wan_va_runtime_server import PROTOCOL


def validate_identity(identity):
    if (identity.get('protocol'), identity.get('model_id'), identity.get('model_revision')) != (
            PROTOCOL, driver.MODEL, driver.REVISION):
        raise ConfigurationError('VA Runtime endpoint protocol/checkpoint mismatch')
    if identity.get('synthetic') is not False or identity.get('seed_mode') != 'episode_plus_frame':
        raise ConfigurationError('requires actual episode-plus-frame inference')
    if identity.get('precision') not in ('native', 'fp8'):
        raise ConfigurationError('missing explicit precision')
    execution = identity.get('execution', {})
    if (execution.get('nfe') != {'video': 20, 'action': 50}
            or execution.get('guidance') != {'video': 5., 'action': 1.}
            or execution.get('grid_shifts') != {'video': 5., 'action': .05}
            or execution.get('dtype') != 'torch.bfloat16'):
        raise ConfigurationError('requires full original LIBERO schedule and BF16 conditioning')
    if execution.get('geometry') != {
            'height': 128, 'width': 128, 'frame_chunk_size': 4, 'action_per_frame': 4,
            'env_type': 'none', 'obs_cam_keys': ['observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb']}:
        raise ConfigurationError('LIBERO geometry mismatch')
    startup = identity.get('startup', {})
    if startup.get('calls') != 6 or startup.get('repeat_byte_equal') is not True or startup.get('action_shape') != [7, 4, 4]:
        raise ConfigurationError('missing completed Runtime startup admission')
    for key in ('checkpoint_sha256', 'runtime_source_sha256', 'engine_source_sha256',
                'upstream_source_sha256', 'pipeline_sha256', 'packages_sha256'):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ConfigurationError(f'missing source/weight identity: {key}')
    if identity['precision'] == 'fp8' and not identity.get('fp8', {}).get('packed_e4m3_tensors'):
        raise ConfigurationError('missing actual FP8 weight admission')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', type=Path, required=True)
    p.add_argument('--identity', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args(argv)
    if not 0 <= a.task < 10 or a.seed < 0 or a.seed == 9173:
        raise ConfigurationError('invalid evaluation scene or reserved startup seed')
    arrays_path = a.output.with_suffix('.npz')
    trace_path = a.output.with_suffix('.trace')
    if a.output.exists() or arrays_path.exists() or trace_path.exists():
        raise ConfigurationError('refusing to overwrite episode evidence')
    identity = load_json(a.identity)
    validate_identity(identity)
    manifest = load_json(a.scenes)
    source = driver.sources(os.environ['LIBERO_ROOT'])
    if manifest.get('protocol') != driver.PROTOCOL or manifest.get('sources') != source:
        raise ConfigurationError('frozen scene simulator source/environment mismatch')
    scene = manifest['scenes'][f'libero_10/{a.task}/{a.seed}']
    if (scene['task'], scene['resolved_seed'], scene['requested_seed']) != (f'libero_10/{a.task}', a.seed, a.seed):
        raise ConfigurationError('frozen scene task/seed mismatch')
    suite, factory = driver.simulator()
    if scene['prompt'] != suite.get_task(a.task).language:
        raise ConfigurationError('frozen task prompt changed')
    env = driver.new_env(suite, factory, a.task, a.seed)
    remote = None
    start = time.monotonic()
    try:
        remote = RemotePolicy(a.endpoint, identity, timeout=300, record_dir=trace_path)
        metrics, actions = driver.rollout_runtime(env, remote, scene)
        import numpy as np
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, actions=np.asarray(actions))
        remote.close()
        timings = remote.timings
        remote = None
        write_json_atomic(a.output, {'ok': True, 'task': a.task, 'seed': a.seed,
            'identity': identity, 'simulator_identity': source, 'scene_sha256': sha256_file(a.scenes),
            'scene': scene, 'scene_content_sha256': sha256_json(scene), 'metrics': metrics,
            'actions_sha256': sha256_file(arrays_path), 'trace_sha256': sha256_file(trace_path / 'trace.json'),
            'driver_sha256': sha256_file(Path(__file__)), 'roundtrips': timings,
            'wall_seconds': time.monotonic() - start,
            'scope': 'Paused observed-history episode; not a standalone quality or real-time certificate'})
    finally:
        if remote is not None:
            remote.close()
        env.close()


if __name__ == '__main__':
    main()
