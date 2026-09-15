"""Pinned LingBot-VA × LIBERO-Long, paused upstream-protocol evaluation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.vla.util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic
from benchmarks.vla.remote_policy import RemotePolicy
from benchmarks.vla.result import validate_result

PROTOCOL = 'wan-va-libero-paused-v1'
MODEL = 'robbyant/lingbot-va-posttrain-libero-long'
REVISION = '0e89d1e753019988aba484e8da2dc0810e264d9f'
SUITE = 'wan_va_libero_long'
EXACT_PASSES = {'fsdp_elision', 'allocator_churn_elision', 'debug_dump_elision', 'conditioning_prefill', 'ring-kv'}


def revision():
    return 'va-libero-v1:' + sha256_json({p.name: sha256_file(p) for p in
        (Path(__file__), Path(__file__).with_name('remote_policy.py'),
         Path(__file__).with_name('adapters.py'), Path(__file__).parent / 'config/adapters.json')})


def sources(root):
    root = Path(root).resolve()
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root/'libero').rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if not files:
        raise ConfigurationError('missing LIBERO source and assets')
    return {'revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'tree_sha256': sha256_json(files),
            'environment': {name: importlib.metadata.version(name) for name in
                ('numpy','scipy','mujoco','robosuite','torch')},
            'render_environment': {name: os.environ.get(name) for name in
                ('MUJOCO_GL','PYOPENGL_PLATFORM','LP_NUM_THREADS','PYTHONHASHSEED')}}


def simulator():
    from libero.libero import benchmark, get_libero_path
    expected = os.environ.get('LIBERO_ROOT')
    if expected:
        base = Path(expected).resolve()/'libero/libero'
        for key, folder in [('bddl_files','bddl_files'),('init_states','init_files'),('assets','assets')]:
            if Path(get_libero_path(key)).resolve() != base/folder:
                raise ConfigurationError('LIBERO config points outside the pinned checkout')
    from libero.libero.envs import OffScreenRenderEnv
    return benchmark.get_benchmark_dict()['libero_10'](), OffScreenRenderEnv


def extract(obs):
    import numpy as np
    return {name: np.ascontiguousarray(obs[key][::-1]) for name,key in (
        ('observation.images.agentview_rgb','agentview_image'),
        ('observation.images.eye_in_hand_rgb','robot0_eye_in_hand_image'))}


def observation_digest(obs):
    import numpy as np
    digest = hashlib.sha256()
    for key, value in sorted(obs.items()):
        a = np.asarray(value)
        if not np.isfinite(a).all() or a.ndim != 3 or not a.any():
            raise ConfigurationError('invalid or blank initial camera observation')
        digest.update(key.encode()); digest.update(str((a.shape,a.dtype.str)).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def initialize(env, state, seed):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    env.seed(seed)
    env.reset()
    env.set_init_state(state)
    # Exact upstream LIBERO VA settling protocol: five all-zero 7D actions.
    for _ in range(5):
        obs, _, _, _ = env.step([0.] * 7)
    return extract(obs)


def new_env(suite, factory, task, seed=0):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    return factory(bddl_file_name=suite.get_task_bddl_file_path(task),camera_heights=128,camera_widths=128)


def prepare(plan, root, output):
    import numpy as np
    from benchmarks.vla.plan import validate_plan
    validate_plan(plan)
    if Path(output).exists():
        raise ConfigurationError('refusing to replace frozen scenes')
    identity = sources(root)
    suite, factory = simulator()
    scenes = {}
    for job in plan['jobs']:
        req = validate_job(job)
        key = scene_key(req)
        if key in scenes: continue
        if req['dataset']['revision'] != identity['revision']:
            raise ConfigurationError('LIBERO revision differs from plan')
        task = int(req['task'].split('/')[-1]); seed = req['requested_seed']
        import torch
        # Official init-state files contain numpy arrays; allow those types locally.
        with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                               np.dtype, type(np.dtype(np.float64))]):
            states = suite.get_task_init_states(task)
        index = seed % len(states)
        state = np.asarray(states[index],dtype=np.float64)
        env = new_env(suite,factory,task,seed)
        try:
            obs = initialize(env,state,seed)
            scenes[key] = {'task':req['task'],'suite_id':SUITE,'requested_seed':seed,'resolved_seed':seed,
                'init_state_index':index,'init_state':state.tolist(),'prompt':suite.get_task(task).language,
                'initial_observation_sha256':observation_digest(obs)}
        finally:
            env.close()
    write_json_atomic(Path(output),{'protocol':PROTOCOL,'sources':identity,'scenes':scenes})


def scene_key(req):
    return f"{req['task']}/{req['requested_seed']}"


def validate_job(job):
    req = job['request']
    if sha256_json(req) != job['request_sha256'] or job['driver']['revision'] != revision():
        raise ConfigurationError('request or driver content changed')
    if req['suite']['id'] != SUITE or req['suite']['protocol'].get('bridge') != PROTOCOL:
        raise ConfigurationError('unsupported LIBERO protocol')
    if (req['model']['checkpoint']['id'], req['model']['checkpoint']['revision']) != (MODEL,REVISION):
        raise ConfigurationError('first LIBERO bridge requires the pinned official checkpoint')
    if req['suite']['seed_strategy'] != 'fixed':
        raise ConfigurationError('LIBERO scenes must use fixed seeds')
    from benchmarks.vla.adapters import validate_bound_adapter
    validate_bound_adapter(req)
    return req


def validate_remote(identity, point):
    if (identity.get('protocol'),identity.get('model_id'),identity.get('model_revision')) != (PROTOCOL,MODEL,REVISION):
        raise ConfigurationError('remote protocol/checkpoint mismatch')
    if identity.get('synthetic') is not False or identity.get('seed_mode') != 'episode_plus_frame':
        raise ConfigurationError('requires a real episode-seeded model')
    execution = identity['execution']
    if execution != point['execution']:
        raise ConfigurationError('remote execution differs from planned point')
    if execution['nfe'] != {'video':20,'action':50} or execution['guidance'] != {'video':5.,'action':1.}:
        raise ConfigurationError('this first campaign preserves original NFE/guidance')
    if execution['grid_shifts'] != {'video':5.,'action':0.05} or execution['dtype'] != 'torch.bfloat16':
        raise ConfigurationError('original scheduler/precision required')
    if not set(execution['applied']) <= EXACT_PASSES:
        raise ConfigurationError('non-exact or unaccounted optimization in strict campaign')
    if execution['geometry'] != {'height':128,'width':128,'frame_chunk_size':4,'action_per_frame':4,
        'env_type':'none','obs_cam_keys':['observation.images.agentview_rgb','observation.images.eye_in_hand_rgb']}:
        raise ConfigurationError('LIBERO observation/action geometry mismatch')


def rollout(env, remote, scene):
    return _rollout(env, remote, scene, runtime_history=False)


def rollout_runtime(env, remote, scene):
    """Drive a Runtime-backed endpoint with observed windows and deferred commit.

    The endpoint must map infer to Runtime.predict and reset_episode to a seeded
    Runtime reset. This helper does not admit an endpoint or certify its identity;
    the legacy run/CLI deliberately retains its separate, strict protocol.
    """
    return _rollout(env, remote, scene, runtime_history=True)


def _rollout(env, remote, scene, *, runtime_history):
    import numpy as np
    first_obs = initialize(env,np.asarray(scene['init_state'],dtype=np.float64),scene['resolved_seed'])
    if observation_digest(first_obs) != scene['initial_observation_sha256']:
        raise ConfigurationError('initial observation differs from frozen scene')
    remote.reset_episode(scene['prompt'],scene['resolved_seed'])
    next_obs = [first_obs] if runtime_history else first_obs
    first, done, cycles = True, False, 0
    digest, trace = hashlib.sha256(), []
    while env.env.timestep < 800:
        result = remote.infer({'obs':next_obs,'prompt':scene['prompt'],'save_visualization':False})
        action = np.asarray(result.get('action'))
        if action.shape != (7,4,4) or not np.isfinite(action).all():
            raise ConfigurationError('expected finite LIBERO VA actions (7,4,4)')
        frames = []
        for i in range(1 if first else 0,4):
            for j in range(4):
                actual = np.ascontiguousarray(action[:,i,j])
                obs, _, done, _ = env.step(actual)
                digest.update(str(actual.dtype).encode()); digest.update(actual.tobytes())
                trace.append(actual.tolist())
                if done: break
                frames.append(extract(obs))
            if done: break
        cycles += 1; first = False
        if done: break  # Upstream does not commit the terminal successful chunk.
        expected = 12 if cycles == 1 else 16
        if len(frames) != expected:
            raise ConfigurationError('incomplete observed history')
        if runtime_history:
            # These frames exist only after the returned actions have executed.
            # Runtime commits its pending action on the next prediction call.
            next_obs = frames
        else:
            remote.infer({'obs':frames,'compute_kv_cache':True,'imagine':False,'state':action,'save_visualization':False})
    return {'success':bool(done),'finite':True,'action_digest':digest.hexdigest(),
            'executed_steps':len(trace),'cycles':cycles},trace


def run(job,root,output):
    req = validate_job(job); point = req['arm']['operating_point']
    ref = point['scene_manifest']; manifest = load_json(Path(ref['path']))
    if sha256_file(Path(ref['path'])) != ref['sha256'] or manifest['protocol'] != PROTOCOL:
        raise ConfigurationError('scene manifest changed')
    identity = sources(root)
    if identity != manifest['sources'] or identity['revision'] != req['dataset']['revision']:
        raise ConfigurationError('LIBERO source/assets differ from campaign')
    scene = manifest['scenes'][scene_key(req)]
    expected_scene = (req['task'],SUITE,req['requested_seed'],req['requested_seed'])
    if tuple(scene[k] for k in ('task','suite_id','requested_seed','resolved_seed')) != expected_scene:
        raise ConfigurationError('scene identity mismatch')
    config = point['remote']; validate_remote(config['identity'],point)
    suite,factory = simulator(); task = int(req['task'].split('/')[-1])
    if suite.get_task(task).language != scene['prompt']:
        raise ConfigurationError('task instruction changed')
    env = new_env(suite,factory,task,scene["resolved_seed"])
    remote = None
    try:
        remote = RemotePolicy(config['endpoint'],config['identity'],timeout=config.get('timeout_seconds',300))
        metrics,trace = rollout(env,remote,scene)
        timing = remote.timings
    finally:
        if remote: remote.close()
        env.close()
    facts = {'sources':identity,'remote':config['identity'],'python':sys.version,
        'packages':sorted(f"{d.metadata.get('Name')}=={d.version}" for d in importlib.metadata.distributions())}
    result = {'schema_version':1,'job_id':job['job_id'],'request_sha256':job['request_sha256'],
        'status':'completed','resolved_seed':scene['resolved_seed'],'metrics':metrics,
        'provenance':{'model_revision':REVISION,'driver_revision':revision(),'synthetic':False,
            'environment_fingerprint':sha256_json(facts),'facts':facts,'scene':scene,'scene_sha256':sha256_json(scene),
            'scene_manifest_sha256':ref['sha256'],'evaluation_mode':'paused_simulation'},
        'diagnostics':{'transport_timings':timing,'claim':'paused quality evaluation; exactness requires paired action bytes'}}
    validate_result(result,job)
    write_json_atomic(Path(output).with_suffix('.actions.json'),trace)
    write_json_atomic(Path(output),result)


def make_plan(arms, output, simulator_root, scenes=None, tasks=10, seeds=2):
    import copy
    from benchmarks.vla.registry import load_registry, Registry, _validate
    from benchmarks.vla.plan import build_plan
    output=Path(output).resolve(); registry_path=output.with_suffix('.registry.json')
    if output.exists() or registry_path.exists(): raise ConfigurationError('use fresh plan paths')
    raw=copy.deepcopy(load_registry().raw)
    raw['models'].append({'id':MODEL,'revision':REVISION,'backbone':'wan_va','builtin':False,
                          'source_env':['LIBERO_ROOT'],'determinism':'bitexact'})
    raw['datasets'].append({'id':'wan_va_libero_sim','kind':'simulator','source':'Lifelong-Robot-Learning/LIBERO',
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=simulator_root,text=True).strip()})
    raw['suites'].append({'id':SUITE,'kind':'closed_loop','dataset':'wan_va_libero_sim',
        'tasks':[f'libero_10/{i}' for i in range(10)],'compatible_backbones':['wan_va'],'model_ids':[MODEL],
        'seed_strategy':'fixed','seed_base':0,'required_metrics':['success','action_digest','finite'],
        'protocol':{'bridge':PROTOCOL,'screening':True,'evaluation_mode':'paused_simulation',
                    'margin':-0.05,'interval':'tango_one_sided95'}})
    raw['profiles']['va_libero_screen']={'suites':[SUITE],'limits':{'tasks':tasks,'seeds_per_task':{'closed_loop':seeds}},
        'arm_repeats':1,'latency':{'warmup':0,'iterations':1}}
    for arm in arms['arms']:
        arm['driver']['revision']=revision()
        if arm['role']=='treatment':
            arm['gates']={'performance':{'min_speedup':1},'action':{'mode':'registry'},
                         'success':{'margin':-0.05,'interval':'tango_one_sided95','min_pairs':100}}
        if scenes:
            arm['operating_point']['scene_manifest']={'path':str(Path(scenes).resolve()),'sha256':sha256_file(Path(scenes))}
        else:
            arm['operating_point'].pop('scene_manifest',None)
    # This profile is screening-only; margin is inherited schema plumbing, not an acceptance budget.
    _validate(raw); registry=Registry(raw,sha256_json(raw),registry_path)
    plan=build_plan(registry,arms,'va_libero_screen',[MODEL])
    for job in plan['jobs']:
        req = validate_job(job)
        if scenes:
            point = req['arm']['operating_point']
            validate_remote(point['remote']['identity'],point)
    if scenes:
        reference = next(a for a in arms['arms'] if a['role']=='control')['operating_point']['remote']['identity']
        if reference['execution']['applied']:
            raise ConfigurationError('control must be the original unoptimized execution')
        for arm in arms['arms']:
            identity = arm['operating_point']['remote']['identity']
            left = {k:v for k,v in reference['execution'].items() if k!='applied'}
            right = {k:v for k,v in identity['execution'].items() if k!='applied'}
            if left != right or identity['checkpoint_sha256'] != reference['checkpoint_sha256']:
                raise ConfigurationError('strict pair changes weights or execution semantics')
    write_json_atomic(registry_path,raw); write_json_atomic(output,plan)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--request',type=Path); mode.add_argument('--prepare-plan',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--libero-root',default=os.environ.get('LIBERO_ROOT'),required=not os.environ.get('LIBERO_ROOT'))
    args=p.parse_args()
    if args.prepare_plan: prepare(load_json(args.prepare_plan),args.libero_root,args.output)
    else: run(load_json(args.request),args.libero_root,args.output)

if __name__=='__main__': main()
