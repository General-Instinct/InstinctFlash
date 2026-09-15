"""Freeze the actual LeRobot reset observation; validate it inside rollout's reset."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import os
from pathlib import Path

from .util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic

PROTOCOL = 'pi05-libero-frozen-scenes-v1'


def observation_digest(value):
    import numpy as np
    def encode(x):
        if isinstance(x, dict): return {k: encode(v) for k, v in sorted(x.items())}
        if x is None or isinstance(x, (str, bool, int, float)): return x
        a = np.asarray(x)
        if a.dtype.kind not in 'biuf' or not a.size or not np.isfinite(a).all():
            raise ConfigurationError('invalid reset observation array')
        return {'shape': list(a.shape), 'dtype': a.dtype.str,
                'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
    pixels = value.get('pixels', {})
    if not isinstance(pixels, dict) or not pixels:
        raise ConfigurationError('missing initial camera pixels')
    for camera in pixels.values():
        a = np.asarray(camera)
        if a.ndim not in (3, 4) or not a.any():
            raise ConfigurationError('blank or malformed initial camera')
    return sha256_json(encode(value))


def identity():
    from .wan_va_libero_driver import sources
    import lerobot
    import transformers
    from libero.libero import get_libero_path
    root = Path(os.environ['LIBERO_ROOT']).resolve()
    for key, folder in [('assets','assets'), ('bddl_files','bddl_files'), ('init_states','init_files')]:
        if Path(get_libero_path(key)).resolve() != root/'libero/libero'/folder:
            raise ConfigurationError('LIBERO configuration differs from pinned root')
    package = Path(lerobot.__file__).resolve().parent
    return {'simulator': sources(root),
            'lerobot_source_sha256': sha256_json({str(p.relative_to(package)): sha256_file(p)
                for p in sorted(package.rglob('*.py'))}),
            'transformers_source_sha256': sha256_json({str(p.relative_to(Path(transformers.__file__).parent)): sha256_file(p)
                for p in sorted(Path(transformers.__file__).parent.rglob('*.py'))}),
            'packages': {k: importlib.metadata.version(k) for k in ('lerobot','gymnasium','torch','numpy','transformers','tokenizers')},
            'source_root': str(root)}


def create_env(suite, task_id, seed):
    from .instinctflash_driver import seed_everything
    from lerobot.envs.factory import make_env, make_env_config
    seed_everything(seed)
    envs = make_env(make_env_config('libero', task=suite, task_ids=[task_id]), n_envs=1, use_async_envs=False)
    vec = envs[suite][task_id]
    vec.envs[0].unwrapped.init_state_id = seed
    return vec


def state_record(vec, seed):
    import numpy as np
    base = vec.envs[0].unwrapped
    states = base._init_states
    if states is None or not len(states): raise ConfigurationError('missing fixed init states')
    index = seed % len(states)
    state = np.asarray(states[index], dtype=np.float64)
    if not state.size or not np.isfinite(state).all(): raise ConfigurationError('invalid init state')
    return {'init_state_index': index, 'init_state': state.tolist(), 'prompt': base.task_description}


def scene_key(suite, task_id, seed):
    return f'{suite}/{task_id}/{seed}'


def prepare(plan, output):
    from .plan import validate_plan
    from .adapters import validate_bound_adapter
    from .pi05_libero_driver import parse_closed_loop_task
    validate_plan(plan)
    if Path(output).exists(): raise ConfigurationError('refusing to overwrite frozen scenes')
    source = identity(); scenes = {}
    for job in plan['jobs']:
        req = job['request']
        if req['suite']['kind'] != 'closed_loop': continue
        if req['dataset']['revision'] != source['simulator']['revision']:
            raise ConfigurationError('plan dataset differs from frozen LIBERO source revision')
        adapter = validate_bound_adapter(req)
        if adapter['id'] != 'pi05-libero-schedule-v1': raise ConfigurationError('requires pi05 LIBERO adapter')
        suite, task = parse_closed_loop_task(req); seed = req['requested_seed']
        key = scene_key(suite, task, seed)
        if key in scenes: continue
        vec = create_env(suite, task, seed)
        try:
            record = state_record(vec, seed)
            obs, _ = vec.reset(seed=[seed])
            scenes[key] = {'suite': suite, 'task_id': task, 'seed': seed,
                           'suite_id': suite, 'task': f'{suite}/{task}',
                           'requested_seed': seed, 'resolved_seed': seed, **record,
                           'initial_observation_sha256': observation_digest(obs)}
        finally: vec.close()
    if not scenes: raise ConfigurationError('no pi05 closed-loop scenes')
    write_json_atomic(Path(output), {'protocol': PROTOCOL, 'sources': source, 'scenes': scenes})


def load_scene(request):
    from .pi05_libero_driver import parse_closed_loop_task
    ref = request['arm']['operating_point'].get('scene_manifest')
    if not ref: raise ConfigurationError('pi05 repeatability requires a frozen scene manifest')
    path = Path(ref['path']); manifest = load_json(path)
    if sha256_file(path) != ref['sha256'] or manifest['protocol'] != PROTOCOL:
        raise ConfigurationError('scene manifest changed')
    if manifest['sources'] != identity(): raise ConfigurationError('scene source/environment changed')
    if request['dataset']['revision'] != manifest['sources']['simulator']['revision']:
        raise ConfigurationError('plan dataset differs from frozen LIBERO source revision')
    suite, task = parse_closed_loop_task(request); seed = request['requested_seed']
    scene = manifest['scenes'][scene_key(suite, task, seed)]
    if (scene['suite'], scene['task_id'], scene['seed']) != (suite, task, seed):
        raise ConfigurationError('scene identity mismatch')
    return scene, manifest['sources'], ref['sha256']


@contextlib.contextmanager
def checked_reset(vec, scene):
    """Intercept the reset that rollout actually consumes; never reset an extra time."""
    expected = {k: scene[k] for k in ('init_state_index', 'init_state', 'prompt')}
    if state_record(vec, scene['seed']) != expected: raise ConfigurationError('frozen initial state/task changed')
    original = vec.reset; calls = []
    def reset(*args, **kwargs):
        if calls or kwargs.get('seed') != [scene['seed']]:
            raise ConfigurationError('unexpected rollout reset or seed')
        result = original(*args, **kwargs)
        if observation_digest(result[0]) != scene['initial_observation_sha256']:
            raise ConfigurationError('initial observation differs from frozen scene')
        calls.append(True)
        return result
    vec.reset = reset
    try:
        yield
        if len(calls) != 1: raise ConfigurationError('rollout did not consume checked reset')
    finally: vec.reset = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(load_json(a.prepare_plan), a.output)


if __name__ == '__main__': main()
