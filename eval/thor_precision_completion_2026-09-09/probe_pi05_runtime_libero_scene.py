"""One frozen LIBERO-10 scene through the public remote Runtime, 50-action queue."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
from benchmarks.vla.remote_policy import RemotePolicy
from benchmarks.vla.wan_va_libero_driver import simulator, sources
p=argparse.ArgumentParser();p.add_argument('--identity',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--endpoint',default='ws://127.0.0.1:19051');p.add_argument('--task',type=int,default=0);p.add_argument('--seed',type=int,default=0);args=p.parse_args()
if args.output.exists():raise RuntimeError('refusing overwrite')
simulator_identity=sources(os.environ['LIBERO_ROOT'])
identity=json.loads(args.identity.read_text());assert identity['protocol']=='pi05-public-libero-50-v1' and identity['n_action_steps']==50
scenes=Path('/home/ubuntu/ifl_eval/libero_va_20260906/scenes-v2.json');scene=json.loads(scenes.read_text())['scenes'][f'libero_10/{args.task}/{args.seed}']
random.seed(args.seed);np.random.seed(args.seed)
suite,factory=simulator();env=factory(bddl_file_name=suite.get_task_bddl_file_path(args.task),camera_heights=360,camera_widths=360)
remote=None;actions=[];success=False;start=time.perf_counter()
keys=['agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
try:
    env.seed(args.seed);env.reset();raw=env.set_init_state(np.asarray(scene['init_state'],dtype=np.float64))
    for _ in range(10):raw,_,_,_=env.step([0,0,0,0,0,0,-1])
    for robot in env.robots:robot.controller.use_delta=True
    initial={k:np.asarray(raw[k]).copy() for k in keys}
    for key in keys[:2]:assert initial[key].shape==(360,360,3) and initial[key].mean()>3
    remote=RemotePolicy(args.endpoint,identity,timeout=180,record_dir=args.output.with_suffix('.trace'))
    remote.reset_episode(scene['prompt'],args.seed)
    for step in range(520):
        reply=remote.infer({'libero_observation':{k:np.asarray(raw[k]) for k in keys},'prompt':scene['prompt']})
        action=np.asarray(reply['action']);assert action.shape==(7,) and np.isfinite(action).all();actions.append(action.copy())
        raw,reward,done,info=env.step(action.tolist());success=bool(env.check_success())
        if step%50==0:print('step',step,'success',success,flush=True)
        if done or success:break
    np.savez_compressed(args.output.with_suffix('.npz'),actions=np.stack(actions),**{'initial_'+k:v for k,v in initial.items()})
    report={'ok':True,'success':success,'steps':len(actions),'task':args.task,'seed':args.seed,'identity':identity,'simulator_identity':simulator_identity,'scene_sha256':hashlib.sha256(scenes.read_bytes()).hexdigest(),'settle_steps':10,'settle_action':[0,0,0,0,0,0,-1],'camera_size':[360,360],'wall_seconds':time.perf_counter()-start,'roundtrips':remote.timings,'scope':'Single-scene closed-loop wiring smoke, native LeRobot environment conversion on server, original 50-action queue; not a success-rate certificate.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ('identity','roundtrips')})
finally:
    env.close()
    if remote is not None:remote.close()
