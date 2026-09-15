"""Run a frozen RoboTwin scene through the observed-history VA Runtime.

The job supplies the existing simulator/scene request only. This produces a new
Runtime episode receipt, not a result for the job's legacy remote operating point.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import re
import time

from . import robotwin_driver as driver
from .remote_policy import RemotePolicy
from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic
from .wan_va_runtime_robotwin_bridge import WanVaRuntimeWireBridge
from .wan_va_runtime_server import PROTOCOL

MODEL = 'robbyant/lingbot-va-posttrain-robotwin'
REVISION = '8c9dea8abbc5c91cc9e18bc3264b8915083bbe70'


@contextmanager
def record_controller_origin(client, output):
    """Record the actual origin passed to native pose composition unchanged."""
    import numpy as np
    original = client.add_init_pose
    origin = None

    def observed(action, initial_pose):
        nonlocal origin
        value = np.asarray(initial_pose)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ConfigurationError('invalid controller origin')
        if origin is None:
            origin = value.copy()
            write_json_atomic(output, {'pose': value.tolist(), 'dtype': value.dtype.str,
                                       'semantics': 'native add_init_pose initial_pose argument'})
        elif (value.dtype != origin.dtype or value.tobytes() != origin.tobytes()):
            raise ConfigurationError('controller origin changed within episode')
        return original(action, initial_pose)

    with driver.patched(client, add_init_pose=observed):
        yield
    if origin is None:
        raise ConfigurationError('episode did not execute native pose composition')


def validate_identity(identity):
    if (identity.get('protocol'), identity.get('model_id'), identity.get('model_revision')) != (
            PROTOCOL, MODEL, REVISION):
        raise ConfigurationError('Runtime RoboTwin protocol/checkpoint mismatch')
    if identity.get('synthetic') is not False or identity.get('seed_mode') != 'episode_plus_frame':
        raise ConfigurationError('requires actual episode-plus-frame inference')
    if identity.get('precision') not in ('native', 'fp8'):
        raise ConfigurationError('missing explicit precision')
    execution = identity.get('execution', {})
    if (execution.get('nfe') != {'video': 25, 'action': 50}
            or execution.get('guidance') != {'video': 5., 'action': 1.}
            or execution.get('grid_shifts') != {'video': 5., 'action': 1.}
            or execution.get('dtype') != 'torch.bfloat16'):
        raise ConfigurationError('requires original RoboTwin schedule and BF16 conditioning')
    if execution.get('geometry') != {
            'height': 256, 'width': 320, 'frame_chunk_size': 2, 'action_per_frame': 16,
            'env_type': 'robotwin_tshape', 'obs_cam_keys': list(driver.CAMERAS)}:
        raise ConfigurationError('RoboTwin geometry mismatch')
    startup = identity.get('startup', {})
    if (startup.get('calls') != 6 or startup.get('repeat_byte_equal') is not True
            or startup.get('action_shape') != [16, 2, 16]):
        raise ConfigurationError('missing completed Runtime startup admission')
    for key in ('checkpoint_sha256', 'runtime_source_sha256', 'engine_source_sha256',
                'upstream_source_sha256', 'pipeline_sha256', 'packages_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(identity.get(key, ''))):
            raise ConfigurationError(f'missing source/weight identity: {key}')
    if identity['precision'] == 'fp8':
        count = identity.get('fp8', {}).get('packed_e4m3_tensors')
        if type(count) is not int or count <= 0:
            raise ConfigurationError('missing actual FP8 weight admission')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'scenes', 'identity', 'output', 'robotwin', 'lingbot'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    a = p.parse_args(argv)
    # Resolve before the upstream simulator changes cwd.
    for name in ('job', 'scenes', 'identity', 'output', 'robotwin', 'lingbot'):
        setattr(a, name, getattr(a, name).resolve())
    work = a.output.with_suffix('.episode')
    if a.output.exists() or work.exists():
        raise ConfigurationError('refusing to overwrite episode evidence')
    request = driver.validate_request(load_json(a.job))
    identity = load_json(a.identity)
    validate_identity(identity)
    checkpoint = request['model']['checkpoint']
    if (checkpoint.get('id'), checkpoint.get('revision')) != (MODEL, REVISION):
        raise ConfigurationError('scene request checkpoint mismatch')
    manifest, scene = driver.load_scene(request, {'path': str(a.scenes), 'sha256': sha256_file(a.scenes)})
    if scene['resolved_seed'] == 9173:
        raise ConfigurationError('reserved startup seed cannot be evaluated')
    sources = driver.source_identity(a.robotwin, a.lingbot)
    if (sources != manifest['sources'] or sources['robotwin_revision'] != request['dataset']['revision']
            or driver.assets_identity(a.robotwin) != manifest['assets_sha256']):
        raise ConfigurationError('simulator source/assets differ from frozen scenes')
    work.mkdir(parents=True)
    client = driver.import_client(a.robotwin, a.lingbot)
    remote = RemotePolicy(a.endpoint, identity, timeout=300, record_dir=work / 'trace')
    start = time.monotonic()
    try:
        with record_controller_origin(client, work / 'controller_origin.json'):
            metrics = driver.rollout(client, request, scene, remote, work,
                                     bridge_factory=WanVaRuntimeWireBridge)
    finally:
        remote.close()
    write_json_atomic(a.output, {
        'ok': True, 'protocol': PROTOCOL, 'identity': identity, 'scene': scene,
        'scene_sha256': sha256_json(scene), 'scene_manifest_sha256': sha256_file(a.scenes),
        'source_request_sha256': sha256_json(request), 'simulator_sources': sources,
        'assets_sha256': manifest['assets_sha256'], 'metrics': metrics,
        'actions_sha256': sha256_file(work / 'executed_actions.json'),
        'controller_origin_sha256': sha256_file(work / 'controller_origin.json'),
        'trace_sha256': sha256_file(work / 'trace' / 'trace.json'),
        'driver_sha256': sha256_file(Path(__file__)), 'roundtrips': remote.timings,
        'wall_seconds': time.monotonic() - start,
        'terminal_history': 'staged locally; no extra terminal prediction or remote commit',
        'scope': 'Paused Runtime episode; requires paired verification; no real-time certificate',
    })


if __name__ == '__main__':
    main()
