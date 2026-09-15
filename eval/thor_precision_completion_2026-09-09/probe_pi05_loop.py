"""Real pi05 processors + Thor engine: queue, state tokens and camera contract."""
import json
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from instinctflash.runtime.pi05_engine import Pi05EngineLoop

out=Path(sys.argv[1])
if out.exists():raise RuntimeError('refusing overwrite')
home=Path.home()
base_model=len(sys.argv)>3 and sys.argv[3]=='base'
model_id='lerobot/pi05_base' if base_model else 'lerobot/pi05_libero_finetuned_v044'
revision='b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba' if base_model else '8e174154ef5f6c60a8da12ae99c303d8963138c1'
checkpoint=home/'.cache/huggingface/hub'/('models--'+model_id.replace('/','--'))/'snapshots'/revision
config=json.loads((checkpoint/'config.json').read_text())
cameras=[k for k in config['input_features'] if k.startswith('observation.images.')]
state_dim=config['input_features']['observation.state']['shape'][0]
action_dim=config['output_features']['action']['shape'][0]
frames=np.load(home/'ifl/t3_assets/calib_obs.npz')
observation={
    cameras[0]: ((frames['image'][3].transpose(2,0,1)+1)/2).astype(np.float32),
    cameras[1]: ((frames['wrist_image'][3].transpose(2,0,1)+1)/2).astype(np.float32),
    'observation.state': np.zeros(state_dim,dtype=np.float32),
}
public_runtime=len(sys.argv)>2 and sys.argv[2]=='runtime'
if public_runtime:
    from instinctflash import Runtime
    api=Runtime.from_pretrained(model_id,
        revision=revision,precision='fp8',device='cuda:0')
    loop=api._backend._loop
else:
    loop=Pi05EngineLoop.from_checkpoint(SimpleNamespace(path=checkpoint),device='cuda:0')
    api=loop
build=loop._frontend_factory
calls=[];tokens=[];raw_chunks=[]
def traced_factory(views):
    fe=build(views);infer=fe.infer;prompt=fe.set_prompt
    def record_prompt(ids):
        tokens.append(np.asarray(ids).copy());return prompt(ids)
    def record_infer(obs):
        calls.append(dict(views=views,images=len(obs['images'])))
        value=infer(obs);raw_chunks.append(value['actions'].copy());return value
    fe.set_prompt=record_prompt;fe.infer=record_infer
    return fe
loop._frontend_factory=traced_factory
api.reset(prompt='pick up the cup')
actions=[]
for i in range(50):
    result=api.predict(observation)
    assert set(result)=={'action'} and result['action'].shape==(action_dim,)
    assert np.isfinite(result['action']).all()
    actions.append(result['action'])
assert len(calls)==1,'action queue unexpectedly regenerated its chunk'
for i,actual in enumerate(actions):
    expected=loop._post(torch.as_tensor(raw_chunks[0][i:i+1,:action_dim],device='cuda')).squeeze(0).cpu().numpy()
    assert np.array_equal(actual,expected),'checkpoint action postprocessor mismatch'
changed={**observation,'observation.state':np.full(state_dim,0.5,dtype=np.float32)}
api.predict(changed)
assert len(calls)==2 and not np.array_equal(tokens[0],tokens[1]),'state was omitted from prompt'
api.reset(prompt='move the cup to the plate')
api.predict(changed)
assert len(calls)==3 and not np.array_equal(tokens[1],tokens[2]),'reset did not update prompt/queue'
api.reset(prompt='pick up the cup')
with_camera={**observation,cameras[2]:np.zeros((3,224,224),dtype=np.float32)}
api.predict(with_camera)
assert calls[-1]['views']==3,'present camera was silently discarded'
api.close()
np.savez(out.with_suffix('.npz'),actions=np.stack(actions),raw_chunks=np.stack(raw_chunks))
out.write_text(json.dumps(dict(checks=dict(native_action_postprocessor=True,single_action_shape=[action_dim],
    computed_horizon=50,buffered_calls_before_refill=50,state_updates_tokens=True,
    episode_reset_updates_prompt_and_queue=True,missing_camera_masked=True,present_camera_preserved=True),
    model_id=model_id,revision=revision,public_runtime=public_runtime,
    actions_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
    loop_source_sha256=hashlib.sha256(Path(inspect.getfile(Pi05EngineLoop)).read_bytes()).hexdigest(),engine_calls=calls,token_lengths=[len(x) for x in tokens],
    scope='Real checkpoint processors and engine contract; recorded processed images plus synthetic state; no simulator quality certificate'),indent=2)+'\n')
print(out.read_text())
