"""Verify default-precision pi05 public API after engine integration changes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from instinctflash import Runtime

out=Path(sys.argv[1]);base=sys.argv[2]=='base'
if out.exists():raise RuntimeError('refusing overwrite')
model='lerobot/pi05_base' if base else 'lerobot/pi05_libero_finetuned_v044'
revision='b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba' if base else '8e174154ef5f6c60a8da12ae99c303d8963138c1'
checkpoint=Path.home()/'.cache/huggingface/hub'/('models--'+model.replace('/','--'))/'snapshots'/revision
cfg=json.loads((checkpoint/'config.json').read_text())
cameras=[k for k in cfg['input_features'] if k.startswith('observation.images.')]
state_dim=cfg['input_features']['observation.state']['shape'][0]
action_dim=cfg['output_features']['action']['shape'][0]
frames=np.load(Path.home()/'ifl/t3_assets/calib_obs.npz')
obs={cameras[0]:((frames['image'][3].transpose(2,0,1)+1)/2).astype(np.float32),
     cameras[1]:((frames['wrist_image'][3].transpose(2,0,1)+1)/2).astype(np.float32),
     'observation.state':np.zeros(state_dim,dtype=np.float32)}
rt=Runtime.from_pretrained(model,revision=revision,device='cuda:0',seed=2718)
rt.reset(prompt='pick up the cup')
policy=rt._backend._impl._p
generate=policy.predict_action_chunk
chunks=[]
def traced(batch,**kwargs):
    value=generate(batch,**kwargs);chunks.append(value.detach().float().cpu().numpy().copy());return value
policy.predict_action_chunk=traced
actions=[]
for i in range(cfg['n_action_steps']):
    result=rt.predict(obs)
    assert set(result)=={'action'} and result['action'].shape==(action_dim,)
    assert np.isfinite(result['action']).all();actions.append(result['action'])
assert len(chunks)==1 and chunks[0].shape==(1,cfg['chunk_size'],action_dim)
changed={**obs,'observation.state':np.full(state_dim,0.5,dtype=np.float32)}
rt.predict(changed);assert len(chunks)==2
rt.reset(prompt='move the cup to the plate');rt.predict(changed);assert len(chunks)==3
dtypes=sorted({str(p.dtype) for p in policy.parameters() if p.is_floating_point()})
assert rt._precision=='native' and all('float8' not in x for x in dtypes)
assert not any(x.name=='engine_offload' and x.applies for x in rt._plan.results)
np.savez(out.with_suffix('.npz'),actions=np.stack(actions),chunks=np.stack(chunks))
result=dict(model_id=model,revision=revision,public_runtime=True,precision='native',
    parameter_dtypes=dtypes,action_dim=action_dim,chunk=cfg['chunk_size'],queue_and_reset_verified=True,
    fp8_not_selected=True,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
    actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
    scope='Default public API contract on recorded processed images and synthetic state, not quality certification')
rt.close();out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
