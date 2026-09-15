"""LeRobot GR00T N1.7 full action-chunk latency using released DROID metadata."""
import argparse,io,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from lerobot.configs import FeatureType,PolicyFeature
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors_from_pretrained
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
report={'framework':'lerobot','family':'groot','calls':[]};actions=[]
try:
 path=snapshot_download('nvidia/GR00T-N1.7-3B',local_files_only=True)
 cameras=['exterior_image_1_left','wrist_image_left']
 features={'observation.images.'+k:PolicyFeature(type=FeatureType.VISUAL,shape=(3,256,256)) for k in cameras}
 features['observation.state']=PolicyFeature(type=FeatureType.STATE,shape=(17,))
 cfg=GrootConfig(base_model_path=path,embodiment_tag='oxe_droid_relative_eef_relative_joint',device='cuda',chunk_size=40,n_action_steps=40,num_inference_timesteps=4,input_features=features,output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(17,))})
 report['config']=cfg.to_dict() if hasattr(cfg,'to_dict') else str(cfg)
 report['checkpoint']=path;report['torch']=torch.__version__;report['fixture_sha256']=hashlib.sha256(a.fixture.read_bytes()).hexdigest()
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(9173)
 start=time.perf_counter();policy=GrootPolicy.from_pretrained(path,config=cfg).eval().to('cuda')
 pre,post=make_groot_pre_post_processors_from_pretrained(cfg,path)
 report['setup_seconds']=time.perf_counter()-start
 data=np.load(a.fixture,allow_pickle=True)
 for i in range(40):
  ims=[np.asarray(Image.open(io.BytesIO(bytes(x))).convert('RGB')).copy() for x in data['jpeg_0'][i%12][:2]]
  state=np.zeros(17,np.float32);state[3:9]=[1,0,0,0,1,0]
  batch={'observation.images.'+k:torch.from_numpy(ims[j]).permute(2,0,1) for j,k in enumerate(cameras)}
  batch.update({'observation.state':torch.from_numpy(state),'task':'pick up the object' if i<8 else 'place the object down'})
  policy.reset();torch.manual_seed(707+i);np.random.seed(707+i)
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():
   action=post(policy.predict_action_chunk(pre(batch)))
  if torch.is_tensor(action):action=action.detach().float().cpu().numpy()
  action=np.asarray(action);torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
  assert action.size and np.isfinite(action).all()
  actions.append(action.copy());row={'i':i,'phase':'warmup' if i<10 else 'measured','ms':ms,'shape':list(action.shape)};report['calls'].append(row);print(row,flush=True)
 samples=[r['ms'] for r in report['calls'] if r['phase']=='measured'];report.update(ok=True,p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),p99_ms=float(np.percentile(samples,99)))
 np.savez_compressed(a.output.with_suffix('.npz'),actions=np.stack(actions))
except Exception as e:report.update(ok=False,error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
finally:a.output.write_text(json.dumps(report,indent=2,default=str)+'\n')
if not report['ok']:raise SystemExit(1)
