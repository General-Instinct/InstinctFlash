"""V2 native processors and reset around the current real FP8 chain on Thor."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from instinctflash.runtime.vla2_engine import build_vla2_engine_loop

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
home=Path.home()
revision='0451855729ec904f970600e0aec8b84661423afe'
model='robbyant/lingbot-vla-v2-6b-robotwin'
path=home/f'.cache/huggingface/hub/models--robbyant--lingbot-vla-v2-6b-robotwin/snapshots/{revision}'
public=len(sys.argv)>2 and sys.argv[2]=='runtime'
if public:
    from instinctflash import Runtime
    api=Runtime.from_pretrained(model,revision=revision,precision='fp8',device='cuda:0')
    loop=api._backend._loop
else:
    ckpt=SimpleNamespace(path=path,model_id=model,execution=SimpleNamespace(extra={
        'robot':'robotwin','checkpoint_subdir':'checkpoints/global_step_50000/hf_ckpt'}))
    loop=build_vla2_engine_loop(ckpt,device='cuda:0');api=loop
frames=np.load(home/'ifl/t3_assets/calib_obs.npz')
def image(key,index):return np.clip((frames[key][index].astype(np.float32)+1)*127.5,0,255).astype(np.uint8)
obs={'observation.images.cam_high':image('image',3),
     'observation.images.cam_left_wrist':image('wrist_image',3),
     'observation.images.cam_right_wrist':image('wrist_image',4),
     'observation.state':np.zeros(14,dtype=np.float32)}
engine=loop._server.vla.model.engine
actions=[]
def predict(observation):
    torch.manual_seed(707)
    result=api.predict(observation)
    assert set(result)=={'action'} and result['action'].shape==(50,14)
    assert np.isfinite(result['action']).all()
    actions.append(result['action'].copy());return actions[-1]
try:
    api.reset(prompt='pick up the cup')
    a=predict(obs);graph=engine._graph
    b=predict(obs)
    assert np.array_equal(a,b),'first/repeated actions differ'
    assert graph is engine._graph,'same prompt recaptured'
    predict({**obs,'observation.state':np.full(14,0.5,dtype=np.float32)})
    assert not np.array_equal(a,actions[-1]),'state ignored'
    predict({**obs,'observation.images.cam_high':image('image',4)})
    assert not np.array_equal(a,actions[-1]),'camera ignored'
    api.reset(prompt='move the cup to the plate')
    predict(obs)
    assert not np.array_equal(a,actions[-1]),'prompt ignored'
    assert graph is not engine._graph,'new prompt retained old graph'
    np.savez(out.with_suffix('.npz'),actions=np.stack(actions))
    result=dict(model_id=model,revision=revision,public_runtime=public,action_shape=[50,14],
                repeat_byte_equal=True,state_camera_prompt_changes=True,reset_verified=True,
                graph_stats=loop.graph_stats,
                artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(inspect.getfile(build_vla2_engine_loop)).read_bytes()).hexdigest(),
                scope='Native robot processors, live BF16 vision and FP8 expert; recorded images/synthetic state; no task-quality certificate')
    out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
finally:api.close()
