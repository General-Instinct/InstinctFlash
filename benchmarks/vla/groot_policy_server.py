"""Original GR00T LIBERO policy / InstinctFlash endpoint with pinned weight identity."""
import argparse,asyncio,importlib.metadata,os,sys
from pathlib import Path
import numpy as np
from benchmarks.vla.util import ConfigurationError,sha256_file,sha256_json,write_json_atomic
from benchmarks.vla.joint_policy_server import Policy
from benchmarks.vla.instinctflash_driver import seed_everything,resolve_snapshot
from benchmarks.vla.plan import pipeline_digest
MODEL='nvidia/GR00T-N1.7-LIBERO'
REVISION='2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21'
KEYS=('x','y','z','roll','pitch','yaw','gripper')
MODES=('stock','runtime_default','runtime_fp8')

def startup_observation(path):
 if path is None:
  obs={f'video.{k}':np.zeros((256,256,3),np.uint8) for k in ('image','wrist_image')}
  obs.update({f'state.{k}':np.zeros(2 if k=='gripper' else 1,np.float32) for k in KEYS})
  return obs
 expected={f'video.{k}':((256,256,3),(np.dtype('uint8'),)) for k in ('image','wrist_image')}
 # Native LiberoEnv returns float64 state; both stock and Runtime cast at the
 # policy-input boundary. Preserve the source bytes in the calibration artifact.
 expected.update({f'state.{k}':((2 if k=='gripper' else 1,),(np.dtype('float32'),np.dtype('float64'))) for k in KEYS})
 with np.load(path,allow_pickle=False) as data:
  if set(data.files)!=set(expected):raise ConfigurationError('startup observation keys differ from native LIBERO inputs')
  obs={k:data[k].copy() for k in data.files}
 for k,(shape,dtypes) in expected.items():
  if obs[k].shape!=shape or obs[k].dtype not in dtypes or not np.isfinite(obs[k]).all():
   raise ConfigurationError(f'invalid startup observation {k}')
 return obs

class Arm:
 def __init__(self,checkpoint,mode):
  if mode not in MODES:raise ConfigurationError(f'unknown GR00T arm {mode!r}')
  self.mode=mode;self.prompt=''
  self.precision='fp8' if mode=='runtime_fp8' else 'native'
  if mode=='stock':
   # Match the upstream workaround used by the Runtime adapter for this Qwen tokenizer.
   import transformers.tokenization_utils_base as tub
   tub.PreTrainedTokenizerBase._patch_mistral_regex=classmethod(lambda cls,tokenizer,*a,**kw:tokenizer)
   from gr00t.policy.gr00t_policy import Gr00tPolicy,Gr00tSimPolicyWrapper
   self.policy=Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag='libero_sim',model_path=str(checkpoint),device='cuda:0',strict=True))
  else:
   from instinctflash import Runtime
   self.policy=Runtime.from_pretrained(str(checkpoint),precision=self.precision,placement='in_process',tier_ceiling='numeric' if self.precision=='fp8' else 'bitexact')
 def precision_receipt(self):
  if self.precision=='native':return {'precision':'native','dtype':'bfloat16'}
  import torch
  loop=self.policy._backend._loop
  weights=loop._frontend._vlsa_q_w
  stats=loop.backend_stats
  if len(weights)!=4 or any(w.dtype!=torch.float8_e4m3fn for w in weights) or loop._runner.replays<1 or stats.get('precision')!='fp8':
   raise ConfigurationError('GR00T FP8 startup did not execute the expected E4M3 VLSA path')
  return {'precision':'fp8','dtype':'mixed','arithmetic':'FP8 VLSA, native BF16 backbone and DiT',
          'startup_vlsa_replays':loop._runner.replays}
 def new_episode(self,prompt):
  self.prompt=prompt
  if self.mode=='stock':self.policy.reset()
  else:self.policy.reset(prompt=prompt)
 def predict(self,obs):
  if self.mode=='stock':
   flat={k:np.asarray(v,dtype=np.uint8 if k.startswith('video.') else np.float32)[None,None] for k,v in obs.items() if k.startswith(('video.','state.'))}
   flat['annotation.human.action.task_description']=[self.prompt]
   actions,_=self.policy.get_action(flat)
   return np.concatenate([actions['action.'+k][0] for k in KEYS],axis=-1)
  output=self.policy.predict(dict(obs,prompt=self.prompt))
  return np.concatenate([output['actions'][k] for k in KEYS],axis=-1)
 def close(self):
  if self.mode!='stock':self.policy.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint-subdir',choices=['libero_10','libero_spatial','libero_object','libero_goal'],default='libero_10');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--receipt',type=Path,required=True)
 p.add_argument('--startup-observation',type=Path,help='NPZ of native flat video/state observations; required for FP8 calibration')
 p.add_argument('--startup-prompt',default='startup shape probe')
 a=p.parse_args()
 if a.receipt.exists():raise ConfigurationError('refusing to overwrite an endpoint receipt')
 if a.mode=='runtime_fp8' and a.startup_observation is None:p.error('runtime_fp8 requires --startup-observation; implicit all-zero calibration is not allowed')
 obs=startup_observation(a.startup_observation)
 source=Path(os.environ['GR00T_ROOT']).resolve();sys.path.insert(0,str(source))
 files={f.name:sha256_file(f) for f in sorted(a.checkpoint.iterdir()) if f.suffix in {'.json','.safetensors'} and f.name!='instinctflash.json'}
 expected=resolve_snapshot(MODEL,REVISION)/a.checkpoint_subdir
 expected_files={f.name:sha256_file(f) for f in sorted(expected.iterdir()) if f.suffix in {'.json','.safetensors'}}
 if files!=expected_files:raise ConfigurationError('checkpoint view differs from pinned official LIBERO fine-tune')
 tree={str(f.relative_to(source)):sha256_file(f) for f in sorted((source/'gr00t').rglob('*.py'))}
 arm=Arm(a.checkpoint,a.mode);seed_everything(0);arm.new_episode(a.startup_prompt)
 result=np.asarray(arm.predict(obs));print('probe',result.shape,flush=True)
 if result.shape!=(16,7) or not np.isfinite(result).all():raise ConfigurationError('unexpected GR00T LIBERO output')
 precision_receipt=arm.precision_receipt()
 precision_receipt['startup_observation_sha256']=sha256_file(a.startup_observation) if a.startup_observation else None
 precision_receipt['startup_prompt']=a.startup_prompt
 import torch
 identity={'schema_version':1,'protocol':'groot-libero-paused-v1','model_id':MODEL,'model_revision':REVISION,'checkpoint_sha256':sha256_json(files),'upstream_sha256':sha256_json(tree),'pipeline_sha256':pipeline_digest(),'seed_mode':'episode','synthetic':False,
 'execution':{'mode':a.mode,'checkpoint_subdir':a.checkpoint_subdir,'action_shape':[16,7],'actions_per_inference':8,'nfe':4,'embodiment':'libero_sim',**precision_receipt,'runtime_explanation':arm.policy.explain() if a.mode!='stock' else None},
 'packages':{k:importlib.metadata.version(k) for k in ('torch','numpy','transformers')},'numeric_environment':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark},'server_sha256':sha256_file(Path(__file__))}
 policy=Policy(arm,identity,seed_everything)
 from instinctflash.serving.msgpack_numpy import Packer,unpackb
 from websockets.asyncio.server import serve
 async def run():
  busy=False
  async def handle(ws):
   nonlocal busy
   if busy:await ws.close(code=1013);return
   busy=True;policy.ready=False;packer=Packer()
   try:
    await ws.send(packer.pack({'benchmark_identity':identity}))
    async for frame in ws:await ws.send(packer.pack(policy.infer(unpackb(frame))))
   except Exception as error:
    try:await ws.send(f'{type(error).__name__}: {error}')
    except Exception:pass
   finally:policy.ready=False;busy=False
  async with serve(handle,'127.0.0.1',a.port,compression=None,max_size=None,ping_interval=None):
   write_json_atomic(a.receipt,identity);print('ready',flush=True);await asyncio.Future()
 try:asyncio.run(run())
 finally:arm.close()
if __name__=='__main__':main()
