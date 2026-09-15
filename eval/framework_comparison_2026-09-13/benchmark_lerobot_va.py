"""LeRobot VA chunk timing with recorded cameras and fixed raw action feedback.

The first chunk omits the conditioning frame in LeRobot's public return value.
Inference still computes two frames; this output-layout difference is recorded.
"""
import argparse,hashlib,io,json,time,traceback,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.policies.factory import make_pre_post_processors
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--video-steps',type=int,default=25);p.add_argument('--action-steps',type=int,default=50);p.add_argument('--episodes',type=int,default=11);a=p.parse_args();assert not a.output.exists()
r={'framework':'lerobot','family':'va','checkpoint':a.checkpoint,'steps':{'video':a.video_steps,'action':a.action_steps},'calls':[],'feedback':'fixed recorded actions, native quantile normalization','first_chunk_return':'16 actions, conditioning frame removed by upstream; later chunks 32','fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest()};outputs={}
try:
 cfg=LingBotVAConfig.from_pretrained(a.checkpoint)
 cfg.device='cuda';cfg.text_encoder_device='cuda';cfg.num_inference_steps=a.video_steps;cfg.action_num_inference_steps=a.action_steps
 # Resolve the exact existing frozen-module snapshot, avoiding an unbound Hub revision.
 cfg.wan_pretrained_path=str(next((Path.home()/'.cache/huggingface/hub/models--robbyant--lingbot-va-posttrain-robotwin/snapshots').iterdir()))
 r['config']=str(cfg);r['torch']=torch.__version__
 policy=LingBotVAPolicy.from_pretrained(a.checkpoint,config=cfg).eval().to('cuda')
 pre,post=make_pre_post_processors(cfg,a.checkpoint,preprocessor_overrides={'device_processor':{'device':'cuda'}})
 stats=load_file(str(Path(a.checkpoint)/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
 q01=torch.zeros(30,1,1);q99=torch.zeros_like(q01)
 ids=cfg.used_action_channel_ids;q01[ids]=stats['action.q01'].reshape(16,1,1);q99[ids]=stats['action.q99'].reshape(16,1,1)
 data=np.load(a.fixture,allow_pickle=True)
 frames=[]
 for i in range(13):
  frames.append({k:torch.from_numpy(np.asarray(Image.open(io.BytesIO(bytes(data['frame0_0'][j] if i==0 else data['jpeg_0'][i-1][j]))).convert('RGB')).copy()).permute(2,0,1).float()/255 for j,k in enumerate(cfg.obs_cam_keys)})
 # Load frozen modules before measuring generation; prompt encoding is episode setup.
 policy._ensure_frozen_modules()
 for i in range(a.episodes*3):
  cycle=i%3
  if cycle==0:
   policy.reset();policy._maybe_init_prompt({'task':'pick up the object' if i<8 else 'place the object down'})
  torch.manual_seed(1300+cycle);np.random.seed(1300+cycle);random.seed(1300+cycle)
  indices=[0] if cycle==0 else list(range(1,5)) if cycle==1 else list(range(5,13))
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():
   batch=pre(dict(frames[0],task='pick up the object' if i<8 else 'place the object down')) if cycle==0 else None
   if cycle:
    policy._obs_buffer=[policy._extract_raw_obs(pre(dict(frames[j]))) for j in indices]
    raw=torch.from_numpy(data['actions_0'][(cycle-1)%len(data['actions_0'])].copy())
    padded=torch.zeros(30,*raw.shape[1:]);padded[ids]=raw
    policy._executed_actions=((padded-q01)/(q99-q01+1e-6)*2-1).unsqueeze(0).unsqueeze(-1)
   action=post(policy.predict_action_chunk(batch)).detach().float().cpu().numpy()
  torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
  assert action.size and np.isfinite(action).all()
  outputs[f'action_{i}']=action;row={'i':i,'cycle':cycle,'phase':'warmup' if i<3 else 'measured','ms':ms,'shape':list(action.shape)};r['calls'].append(row);print(row,flush=True)
 samples=[x['ms'] for x in r['calls'] if x['phase']=='measured'];r['ok']=True
 if samples:r.update(p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),p99_ms=float(np.percentile(samples,99)))
 else:r['smoke_only']=True
 np.savez_compressed(a.output.with_suffix('.npz'),**outputs)
except Exception as e:r.update(ok=False,error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:a.output.write_text(json.dumps(r,indent=2)+'\n')
if not r['ok']:raise SystemExit(1)
