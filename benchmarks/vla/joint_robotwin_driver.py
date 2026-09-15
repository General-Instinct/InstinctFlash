"""LingBot-VLA 4B/V2 joint-action closed-loop evaluation on frozen RoboTwin scenes."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import os
import struct
import sys
import tempfile

from . import robotwin_driver as sim
from .adapters import validate_bound_adapter
from .remote_policy import RemotePolicy
from .result import validate_result
from .util import ConfigurationError,load_json,sha256_file,sha256_json,write_json_atomic

PROTOCOL='lingbot-joint-robotwin-scenes-v1'


def revision():
    return 'joint-robotwin-v1:'+sha256_json({p.name:sha256_file(p) for p in
        (Path(__file__),Path(sim.__file__),Path(__file__).with_name('remote_policy.py'),
         Path(__file__).with_name('adapters.py'),Path(__file__).parent/'config/adapters.json')})


def validate_job(job):
    r=job['request']
    if job['request_sha256'] != sha256_json(r) or job['driver']['revision'] != revision():
        raise ConfigurationError('changed request or driver; rebuild plan')
    contract=validate_bound_adapter(r)
    if contract['driver_module'] != __name__ and contract['driver_module'] != 'benchmarks.vla.joint_robotwin_driver':
        raise ConfigurationError('wrong joint policy adapter')
    if r['suite']['id'] not in sim.SETTINGS or r['suite']['protocol'].get('setting') != sim.SETTINGS[r['suite']['id']]:
        raise ConfigurationError('RoboTwin setting mismatch')
    return r,contract


def rollout(env,settings,scene,remote,shape):
    import numpy as np
    settings=dict(settings,eval_mode=True,render_freq=0,eval_video_log=False)
    settings.pop('eval_video_save_dir',None)
    env.setup_demo(now_ep_num=0,seed=scene['resolved_seed'],is_test=True,**settings)
    if sim.initial_state(env.get_obs()) != scene['initial_state_sha256']:
        raise ConfigurationError('initial camera/state differs from frozen scene')
    env.set_instruction(instruction=scene['prompt'])
    remote.reset_episode(scene['prompt'],scene['resolved_seed'])
    digest=hashlib.sha256();trace=[]
    while env.take_action_cnt < env.step_lim and not env.eval_success:
        obs=env.get_obs()
        request={f'observation.images.{key}':obs['observation'][camera]['rgb'] for key,camera in
                 [('cam_high','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]}
        request.update({'observation.state':obs['joint_action']['vector'],'task':scene['prompt']})
        actions=np.asarray(remote.infer(request)['action'])
        if tuple(actions.shape)!=tuple(shape) or not np.isfinite(actions).all():
            raise ConfigurationError('invalid joint action chunk')
        for action in actions:
            if env.take_action_cnt>=env.step_lim or env.eval_success:break
            before=env.take_action_cnt
            env.take_action(action,action_type='qpos')
            if env.take_action_cnt != before+1:raise ConfigurationError('controller did not execute exactly one action')
            values=np.asarray(action,dtype=np.float64).tolist()
            digest.update(b''.join(struct.pack('!d',v) for v in values));trace.extend(values)
    if not trace:raise ConfigurationError('empty action trajectory')
    return {'success':bool(env.eval_success),'finite':True,'executed_steps':len(trace)//14,
            'action_digest':digest.hexdigest(),'action_values':trace}


def run(job,robotwin,lingbot,output):
    r,contract=validate_job(job);point=r['arm']['operating_point']
    manifest,scene=sim.load_scene(r,point['scene_manifest'],protocol=PROTOCOL)
    sources=sim.source_identity(robotwin,lingbot)
    if sources!=manifest['sources'] or sources['robotwin_revision']!=r['dataset']['revision']:
        raise ConfigurationError('changed simulator source')
    if sim.assets_identity(robotwin)!=manifest['assets_sha256']:raise ConfigurationError('changed simulator assets')
    identity=point['remote']['identity'];checkpoint=r['model']['checkpoint']
    if (identity['model_id'],identity['model_revision'])!=(checkpoint['id'],checkpoint['revision']):
        raise ConfigurationError('remote checkpoint mismatch')
    if identity['protocol']!=contract['protocol'] or identity['seed_mode']!='episode' or identity['synthetic'] is not False:
        raise ConfigurationError('unsupported remote protocol')
    if identity['execution']!=point['execution'] or identity['execution']['action_shape']!=contract['actions']['wire_shape']:
        raise ConfigurationError('remote execution mismatch')
    client=sim.import_client(robotwin,lingbot)
    remote=RemotePolicy(point['remote']['endpoint'],identity,timeout=300,
                        record_dir=Path(output).with_suffix('.trace') if point.get('record_inputs') else None)
    try:
        with tempfile.TemporaryDirectory(prefix='ifl-joint-') as work:
            metrics=sim.with_upstream_setup(client,r,lambda env,args,kw:rollout(env,args,scene,remote,contract['actions']['wire_shape']),work)
    finally:remote.close()
    facts={'sources':sources,'assets_sha256':manifest['assets_sha256'],'remote':identity,
           'packages':sorted(f"{d.metadata.get('Name')}=={d.version}" for d in importlib.metadata.distributions())}
    result={'schema_version':1,'job_id':job['job_id'],'request_sha256':job['request_sha256'],'status':'completed',
        'resolved_seed':scene['resolved_seed'],'metrics':metrics,'provenance':{'model_revision':checkpoint['revision'],
        'driver_revision':revision(),'synthetic':False,'environment_fingerprint':sha256_json(facts),
        'scene':scene,'scene_sha256':sha256_json(scene),'scene_manifest_sha256':point['scene_manifest']['sha256'],
        'evaluation_mode':'paused_simulation','facts':facts},'diagnostics':{'transport_timings':remote.timings}}
    validate_result(result,job);write_json_atomic(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--request',type=Path);mode.add_argument('--prepare-plan',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();robotwin=Path(os.environ['ROBOTWIN_ROOT']).resolve();lingbot=Path(os.environ['LINGBOT_ROOT']).resolve()
    if a.request:run(load_json(a.request),robotwin,lingbot,a.output);return
    from .plan import validate_plan
    plan=load_json(a.prepare_plan);validate_plan(plan)
    if a.output.exists():raise ConfigurationError('refusing to overwrite frozen scenes')
    requests={}
    sources=sim.source_identity(robotwin,lingbot)
    for job in plan['jobs']:
        r,_=validate_job(job)
        if r['dataset']['revision']!=sources['robotwin_revision']:raise ConfigurationError('dataset revision mismatch')
        requests[sim.scene_key(r)]=r
    assets=sim.assets_identity(robotwin);client=sim.import_client(robotwin,lingbot);scenes={}
    with tempfile.TemporaryDirectory(prefix='ifl-joint-scenes-') as work:
        for key,r in requests.items():
            used={v['resolved_seed'] for v in scenes.values() if v['task']==r['task'] and v['suite_id']==r['suite']['id']}
            scenes[key]=sim.prepare_scene(client,r,work,used)
    write_json_atomic(a.output,{'schema_version':1,'protocol':PROTOCOL,'reset_policy':sim.RESET_POLICY,'sources':sources,'assets_sha256':assets,'scenes':scenes})

if __name__=='__main__':main()
