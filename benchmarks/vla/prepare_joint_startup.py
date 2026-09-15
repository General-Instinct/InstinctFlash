"""Freeze a held-out native RoboTwin reset for joint-policy startup/calibration."""
import argparse
import copy
from pathlib import Path
import tempfile

import numpy as np

from . import robotwin_driver as sim
from .joint_policy_server import load_startup_observation
from .util import ConfigurationError, load_json, sha256_file, write_json_atomic


def prepare(robotwin, lingbot, request, output, seed):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = copy.deepcopy(request)
    request['requested_seed'] = seed
    request['suite']['seed_strategy'] = 'fixed'
    sources = sim.source_identity(robotwin, lingbot)
    assets = sim.assets_identity(robotwin)
    client = sim.import_client(robotwin, lingbot)
    snapshots = []
    with tempfile.TemporaryDirectory(prefix='ifl-joint-startup-') as work:
        scene = sim.prepare_scene(client, request, work)
        def capture(env, args, kwargs):
            settings = dict(args, eval_mode=True, render_freq=0, eval_video_log=False)
            settings.pop('eval_video_save_dir', None)
            env.setup_demo(now_ep_num=0, seed=scene['resolved_seed'], is_test=True, **settings)
            obs = env.get_obs()
            if sim.initial_state(obs) != scene['initial_state_sha256']:
                raise ConfigurationError('startup reset differs from frozen expert scene')
            arrays = {f'observation.images.{key}': np.array(obs['observation'][camera]['rgb'], copy=True)
                      for key, camera in [('cam_high','head_camera'),
                                          ('cam_left_wrist','left_camera'),
                                          ('cam_right_wrist','right_camera')]}
            arrays['observation.state'] = np.array(obs['joint_action']['vector'], copy=True)
            snapshots.append(arrays)
            return {'success': True}
        for _ in range(2):
            sim.with_upstream_setup(client, request, capture, work)
    path = output/'observation.npz'
    np.savez(path, **snapshots[0])
    loaded = load_startup_observation(path, scene['prompt'])
    for key, value in snapshots[1].items():
        if loaded[key].dtype != value.dtype or loaded[key].shape != value.shape or loaded[key].tobytes() != value.tobytes():
            raise ConfigurationError('startup observation reset/serialization is not byte-repeatable')
    if sources != sim.source_identity(robotwin, lingbot):
        raise ConfigurationError('simulator sources changed during preparation')
    report = {'purpose': 'startup/calibration only; exclude this task/seed from evaluation',
              'scene': scene, 'sources': sources, 'assets_sha256': assets,
              'observation_sha256': sha256_file(path), 'repeated_native_reset_byte_equal': True,
              'preparation_source_sha256': sha256_file(Path(__file__)),
              'arrays': {k: {'shape': list(v.shape), 'dtype': v.dtype.str} for k,v in snapshots[0].items()}}
    write_json_atomic(output/'receipt.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robotwin-root', type=Path, required=True)
    parser.add_argument('--lingbot-root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True, help='request template; no policy is connected')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=9173)
    args = parser.parse_args()
    prepare(args.robotwin_root.resolve(), args.lingbot_root.resolve(), load_json(args.request),
            args.output, args.seed)
