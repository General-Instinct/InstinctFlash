"""Native raw-camera GR00T policy with actual FP8 VLSA and BF16 DiT."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from instinctflash.runtime.groot_engine import build_groot_engine_loop

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
checkpoint=Path.home()/'.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495'
ckpt=SimpleNamespace(path=checkpoint,model_id='nvidia/GR00T-N1.7-3B',execution=SimpleNamespace(extra={}))
mode=sys.argv[2] if len(sys.argv)>2 else 'direct'
public=mode in {'runtime','native'}
if public:
    from instinctflash import Runtime
    api=Runtime.from_pretrained('nvidia/GR00T-N1.7-3B',revision='2fc962b973bccdd5d8ce4f67cc63b264d6886495',
                               precision='native' if mode=='native' else 'fp8',device='cuda:0')
    api.reset(prompt='pick up the cup')
    loop=api._backend._impl if mode=='native' else api._backend._loop
else:
    loop=build_groot_engine_loop(ckpt,device='cuda:0');api=loop
frames=np.load(Path.home()/'ifl/t3_assets/calib_obs.npz')
def image(key,index):return np.clip((frames[key][index].astype(np.float32)+1)*127.5,0,255).astype(np.uint8)
state={key:np.zeros(dim,dtype=np.float32) for key,dim in loop._state_dims.items()}
for key,value in state.items():
    if key.endswith('eef_9d'):value[3:9]=[1,0,0,0,1,0]
obs={'images':[image('image',3),image('wrist_image',3)],'state':state}
backbone_calls=[]
handle=loop._policy.model.backbone.register_forward_hook(lambda *args:backbone_calls.append(1))
actions=[]
def predict(observation):
    torch.manual_seed(191)
    result=api.predict(observation)
    assert set(result)=={'action','actions','info'}
    assert result['action'].shape[0]==40 and np.isfinite(result['action']).all()
    actions.append(result['action'].copy());return actions[-1]
try:
    api.reset(prompt='pick up the cup')
    a=predict(obs);b=predict(obs)
    assert np.array_equal(a,b),'fixed-input first/repeated output differs'
    predict({**obs,'state':{k:v+0.1 for k,v in state.items()}})
    assert not np.array_equal(a,actions[-1]),'state ignored'
    predict({**obs,'images':[image('image',4),image('wrist_image',3)]})
    assert not np.array_equal(a,actions[-1]),'camera ignored'
    api.reset(prompt='move the cup to the plate');predict(obs)
    assert not np.array_equal(a,actions[-1]),'prompt ignored'
    if mode!='native':
        assert len(backbone_calls)==len(actions),'native backbone was cached across observations'
    parameter_dtypes=sorted({str(p.dtype) for p in loop._policy.model.parameters()})
    if mode=='native':assert not any('float8' in d for d in parameter_dtypes)
    np.savez(out.with_suffix('.npz'),actions=np.stack(actions))
    report=dict(action_shape=list(a.shape),backbone_calls=len(backbone_calls),
        repeat_byte_equal=True,state_camera_prompt_changes=True,reset_verified=True,
        backend_stats=loop.backend_stats,public_runtime=public,
        precision='native' if mode=='native' else 'fp8',
        native_policy_parameter_dtypes=parameter_dtypes,
        source_sha256=hashlib.sha256(Path(inspect.getfile(type(loop) if mode=='native' else build_groot_engine_loop)).read_bytes()).hexdigest(),
        artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
        scope='Public/native-processor action contract on recorded images/synthetic states; arithmetic in precision/backend_stats; no paired quality certificate')
finally:
    head=loop._policy.model.action_head
    handle.remove();api.close()
    if mode!='native':assert 'get_action' not in head.__dict__,'close retained action-generator hook'
report['close_restores_native_action_head']=mode!='native'
out.write_text(json.dumps(report,indent=2)+'\n');print(out.read_text())
