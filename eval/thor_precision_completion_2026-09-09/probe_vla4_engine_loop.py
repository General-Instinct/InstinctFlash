"""Run the VLA4 native robot processors around the actual Thor FP8 generator."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from instinctflash.runtime.vla4_engine import build_vla4_engine_loop

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
home=Path.home()
checkpoint=home/'.cache/huggingface/hub/models--robbyant--lingbot-vla-4b-posttrain-robotwin/snapshots/fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e'
ckpt=SimpleNamespace(path=checkpoint,execution=SimpleNamespace(extra={
    'robot':'robotwin','use_length':25,'norm_stats':'assets/norm_stats/robotwin_50.json'}))
public_runtime=len(sys.argv)>2 and sys.argv[2]=='runtime'
if public_runtime:
    from instinctflash import Runtime
    api=Runtime.from_pretrained('robbyant/lingbot-vla-4b-posttrain-robotwin',
        revision='fb71a2c9749ccfedbb7290c2c3f0e5e7c7305c9e',precision='fp8',device='cuda:0')
    loop=api._backend._loop
else:
    loop=build_vla4_engine_loop(ckpt,device='cuda:0');api=loop
frames=np.load(home/'ifl/t3_assets/calib_obs.npz')
def image(key,index):
    return np.clip((frames[key][index].astype(np.float32)+1)*127.5,0,255).astype(np.uint8)
obs={'observation.images.cam_high':image('image',3),
     'observation.images.cam_left_wrist':image('wrist_image',3),
     'observation.images.cam_right_wrist':image('wrist_image',4),
     'observation.state':np.zeros(14,dtype=np.float32)}
fe=loop._server.vla.model.frontend
infer=fe.infer_staged;calls=[];staged=[];raw=[];visual=[];keys=[]
def trace(patches,state,noise,**kwargs):
    calls.append(dict(patch_shape=list(patches.shape),state_shape=list(state.shape),
                      noise_shape=list(noise.shape)))
    staged.append(patches.detach().float().cpu().numpy().copy())
    result=infer(patches,state,noise,**kwargs)
    raw.append(result['actions'].copy())
    visual.append(fe._vis_emb.float().cpu().numpy().copy())
    keys.append(fe._Kc[:,:192].float().cpu().numpy().copy())
    print('refresh diagnostic',len(raw),
          'vision delta',float(np.max(np.abs(visual[-1]-visual[0]))),
          'prefix KV delta',float(np.max(np.abs(keys[-1]-keys[0]))),
          'raw delta',float(np.max(np.abs(raw[-1]-raw[0]))),flush=True)
    np.savez(out.with_suffix('.debug.npz'),patches=np.stack(staged),raw=np.stack(raw),visual=np.stack(visual),keys=np.stack(keys))
    return result
fe.infer_staged=trace
api.reset(prompt='pick up the cup')
actions=[]
def predict(observation):
    torch.manual_seed(707)
    result=api.predict(observation)
    assert set(result)=={'action'} and result['action'].shape==(25,14)
    assert np.isfinite(result['action']).all()
    actions.append(result['action'].copy());return result['action']
a=predict(obs);graph=fe._g_lm
b=predict(obs)
assert np.array_equal(a,b),'fixed-input first/repeated replies differ'
assert graph is fe._g_lm,'unchanged prompt unnecessarily recaptured'
predict({**obs,'observation.state':np.full(14,0.5,dtype=np.float32)})
assert not np.array_equal(actions[-1],a),'changed state was ignored'
predict({**obs,'observation.images.cam_high':image('image',4)})
assert not np.array_equal(actions[-1],a),f'changed camera reply identical; patch delta={np.max(np.abs(staged[-1]-staged[0]))}, raw delta={np.max(np.abs(raw[-1]-raw[0]))}'
api.reset(prompt='move the cup to the plate')
predict(obs)
assert not np.array_equal(actions[-1],a),'reset prompt was ignored'
np.savez(out.with_suffix('.npz'),actions=np.stack(actions))
result=dict(model_id='robbyant/lingbot-vla-4b-posttrain-robotwin',public_runtime=public_runtime,
            native_processor_loop=True,fp8_generator=type(fe).__name__,
            action_shape=[25,14],same_input_byte_equal=True,state_camera_prompt_live=True,
            engine_calls=calls,
            bridge_sha256=hashlib.sha256(Path(inspect.getfile(build_vla4_engine_loop)).read_bytes()).hexdigest(),
            actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
            scope='Native robot processors and FP8 execution on recorded images with synthetic state; no simulator quality certificate')
api.close();out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
