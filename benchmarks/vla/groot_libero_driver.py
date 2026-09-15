"""GR00T N1.7 LIBERO-10 native reset protocol, frozen initial observations."""
import argparse,hashlib,importlib.metadata,os,sys,random,subprocess
from pathlib import Path
from .util import ConfigurationError,load_json,sha256_file,sha256_json,write_json_atomic
from .remote_policy import RemotePolicy
from .adapters import validate_bound_adapter
from .result import validate_result
from .wan_va_libero_driver import sources
PROTOCOL='groot-libero-paused-v1'
MODEL='nvidia/GR00T-N1.7-LIBERO'
REVISION='2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21'
SUITE='groot_libero_10'
KEYS=('x','y','z','roll','pitch','yaw','gripper')

def revision():
 return 'groot-libero-v1:'+sha256_json({p.name:sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('remote_policy.py'),Path(__file__).with_name('wan_va_libero_driver.py'),Path(__file__).with_name('adapters.py'),Path(__file__).parent/'config/adapters.json')})

def source_identity(root,groot):
 wrapper=load_groot_wrapper(groot)
 validate_libero_binding(root)
 result=sources(root)
 result['groot_wrapper_sha256']=sha256_file(Path(wrapper.__file__))
 return result

def validate_libero_binding(root):
 from libero.libero import benchmark,get_libero_path
 base=Path(root).resolve()/'libero/libero'
 if not Path(benchmark.__file__).resolve().is_relative_to(base):
  raise ConfigurationError('imported LIBERO code differs from declared checkout')
 for key,folder in [('bddl_files','bddl_files'),('init_states','init_files'),('assets','assets')]:
  if Path(get_libero_path(key)).resolve()!=base/folder:
   raise ConfigurationError(f'LIBERO {key} points outside declared checkout')

def load_groot_wrapper(groot):
 import importlib
 sys.path.insert(0,str(Path(groot).resolve()))
 wrapper=importlib.import_module('gr00t.eval.sim.LIBERO.libero_env')
 expected=Path(groot).resolve()/'gr00t/eval/sim/LIBERO/libero_env.py'
 if Path(wrapper.__file__).resolve()!=expected:
  raise ConfigurationError('imported GR00T simulator wrapper differs from declared checkout')
 return wrapper

def simulator(groot,suite_name="libero_10"):
 wrapper=load_groot_wrapper(groot)
 from libero.libero import benchmark
 return benchmark.get_benchmark_dict()[suite_name](),wrapper.LiberoEnv

def observation_digest(obs):
 import numpy as np
 h=hashlib.sha256()
 for k,v in sorted(obs.items()):
  if isinstance(v,str):h.update(k.encode());h.update(v.encode());continue
  a=np.ascontiguousarray(v)
  if not np.isfinite(a).all() or (k.startswith('video.') and not a.any()):raise ConfigurationError('invalid initial observation')
  h.update(k.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
 return h.hexdigest()

def new_env(suite,factory,task,seed):
 import numpy as np
 random.seed(seed);np.random.seed(seed)
 return factory(suite.get_task_bddl_file_path(task),suite.get_task(task).language)

def reset(env,seed):
 import numpy as np
 random.seed(seed);np.random.seed(seed)
 return env.reset(seed=seed)[0]

def validate_job(job):
 r=job['request']
 if sha256_json(r)!=job['request_sha256'] or job['driver']['revision']!=revision():raise ConfigurationError('request or driver changed')
 validate_bound_adapter(r)
 if r['suite']['protocol'].get('bridge')!=PROTOCOL:raise ConfigurationError('wrong GR00T suite')
 if r['task'].split('/')[0]!=r['model']['checkpoint'].get('subdir','libero_10'):raise ConfigurationError('task and checkpoint subdirectory differ')
 if (r['model']['checkpoint']['id'],r['model']['checkpoint']['revision'])!=(MODEL,REVISION):raise ConfigurationError('requires pinned LIBERO fine-tune')
 if r['suite']['seed_strategy']!='fixed':raise ConfigurationError('fixed seeds required')
 return r

def scene_key(r):return f"{r['task']}/{r['requested_seed']}"

def prepare(plan,root,groot,output):
 from .plan import validate_plan
 validate_plan(plan)
 if Path(output).exists():raise ConfigurationError('scene manifest already exists')
 identity=source_identity(root,groot);suite,factory=simulator(groot,plan['jobs'][0]['request']['model']['checkpoint'].get('subdir','libero_10'));scenes={}
 for job in plan['jobs']:
  r=validate_job(job);key=scene_key(r)
  if key in scenes:continue
  if r['dataset']['revision']!=identity['revision']:raise ConfigurationError('LIBERO revision mismatch')
  task=int(r['task'].split('/')[-1]);seed=r['requested_seed'];env=new_env(suite,factory,task,seed)
  try:
   obs=reset(env,seed)
   scenes[key]={'task':r['task'],'suite_id':r['suite']['id'],'requested_seed':seed,'resolved_seed':seed,'prompt':suite.get_task(task).language,'initial_observation_sha256':observation_digest(obs),'reset_protocol':'official GR00T LiberoEnv.reset(seed); no extra settling or init-state replacement'}
  finally:env.close()
 write_json_atomic(output,{'protocol':PROTOCOL,'sources':identity,'scenes':scenes})

def rollout(env,remote,scene):
 import numpy as np
 obs=reset(env,scene['resolved_seed'])
 if observation_digest(obs)!=scene['initial_observation_sha256']:raise ConfigurationError('initial observation mismatch')
 remote.reset_episode(scene['prompt'],scene['resolved_seed']);trace=[];success=False
 # Instrument the actual upstream controller call, after native gripper conversion.
 original=env._env.step
 def recorded(action):
  value=np.asarray(action)
  if value.shape!=(7,) or not np.isfinite(value).all():raise ConfigurationError('invalid executed controller action')
  result=original(action);trace.extend(np.asarray(value,dtype=np.float64).tolist());return result
 env._env.step=recorded
 try:
  while len(trace)//7<720 and not success:
   inputs={k:np.ascontiguousarray(v) for k,v in obs.items() if k.startswith(('video.','state.'))}
   chunk=np.asarray(remote.infer(inputs)['action'])
   if chunk.shape!=(16,7) or not np.isfinite(chunk).all():raise ConfigurationError('expected finite 16x7 action chunk')
   for action in chunk[:8]:
    obs,_,done,_,info=env.step({'action.'+k:action[i:i+1] for i,k in enumerate(KEYS)})
    success=bool(info['success'])
    if success or done or len(trace)//7>=720:break
   if done:break
 finally:env._env.step=original
 if not trace:raise ConfigurationError('empty trajectory')
 return {'success':success,'finite':True,'executed_steps':len(trace)//7,'action_digest':hashlib.sha256(np.asarray(trace,dtype='>f8').tobytes()).hexdigest(),'action_values':trace}

def run(job,root,groot,output):
 r=validate_job(job);point=r['arm']['operating_point'];ref=point['scene_manifest'];manifest=load_json(Path(ref['path']))
 if sha256_file(Path(ref['path']))!=ref['sha256'] or manifest['protocol']!=PROTOCOL:raise ConfigurationError('changed manifest')
 identity=source_identity(root,groot)
 if identity!=manifest['sources'] or identity['revision']!=r['dataset']['revision']:raise ConfigurationError('changed simulator sources/assets')
 scene=manifest['scenes'][scene_key(r)];config=point['remote'];remote_id=config['identity']
 if (remote_id.get('protocol'),remote_id.get('model_id'),remote_id.get('model_revision'))!=(PROTOCOL,MODEL,REVISION):raise ConfigurationError('wrong policy identity')
 if remote_id.get('seed_mode')!='episode' or remote_id.get('synthetic') is not False or remote_id['execution']!=point['execution']:raise ConfigurationError('wrong policy execution')
 if remote_id['execution'].get('checkpoint_subdir','libero_10')!=r['model']['checkpoint'].get('subdir','libero_10'):raise ConfigurationError('policy checkpoint subdirectory mismatch')
 if remote_id['execution']['nfe']!=4 or remote_id['execution']['actions_per_inference']!=8:raise ConfigurationError('changed native action schedule')
 suite,factory=simulator(groot,r['model']['checkpoint'].get('subdir','libero_10'));task=int(r['task'].split('/')[-1]);env=new_env(suite,factory,task,scene['resolved_seed']);remote=None
 try:
  remote=RemotePolicy(config['endpoint'],remote_id,timeout=300);metrics=rollout(env,remote,scene);timings=remote.timings
 finally:
  if remote:remote.close()
  env.close()
 facts={'sources':identity,'remote':remote_id}
 result={'schema_version':1,'job_id':job['job_id'],'request_sha256':job['request_sha256'],'status':'completed','resolved_seed':scene['resolved_seed'],'metrics':metrics,'provenance':{'model_revision':REVISION,'driver_revision':revision(),'synthetic':False,'environment_fingerprint':sha256_json(facts),'facts':facts,'scene':scene,'scene_sha256':sha256_json(scene),'scene_manifest_sha256':ref['sha256'],'evaluation_mode':'paused_simulation'},'diagnostics':{'transport_timings':timings}}
 validate_result(result,job);write_json_atomic(output,result)

def main():
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True);m.add_argument('--request',type=Path);m.add_argument('--prepare-plan',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--libero-root',default=os.environ.get('LIBERO_ROOT'));p.add_argument('--groot-root',default=os.environ.get('GR00T_ROOT'));a=p.parse_args()
 if not a.libero_root or not a.groot_root:p.error('LIBERO_ROOT and GR00T_ROOT required')
 if a.prepare_plan:prepare(load_json(a.prepare_plan),a.libero_root,a.groot_root,a.output)
 else:run(load_json(a.request),a.libero_root,a.groot_root,a.output)
if __name__=='__main__':main()
