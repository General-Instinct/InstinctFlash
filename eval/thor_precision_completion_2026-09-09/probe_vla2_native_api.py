"""Current default V2 public Runtime contract and actual arithmetic on Thor."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from instinctflash import Runtime

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
model='robbyant/lingbot-vla-v2-6b-robotwin';revision='0451855729ec904f970600e0aec8b84661423afe'
t0=time.perf_counter()
api=Runtime.from_pretrained(model,revision=revision,precision='native',device='cuda:0')
api.reset(prompt='pick up the cup')
setup=time.perf_counter()-t0
loop=api._backend._impl
dtypes=sorted({str(p.dtype) for p in loop._server.vla.parameters()})
assert not any('float8' in d for d in dtypes)
frames=np.load(Path.home()/'ifl/t3_assets/calib_obs.npz')
def image(key,index):return np.clip((frames[key][index].astype(np.float32)+1)*127.5,0,255).astype(np.uint8)
obs={'observation.images.cam_high':image('image',3),
     'observation.images.cam_left_wrist':image('wrist_image',3),
     'observation.images.cam_right_wrist':image('wrist_image',4),
     'observation.state':np.zeros(14,dtype=np.float32)}
actions=[];latencies=[]
def predict(observation):
    torch.manual_seed(707);torch.cuda.synchronize();t0=time.perf_counter()
    result=api.predict(observation);torch.cuda.synchronize()
    latencies.append((time.perf_counter()-t0)*1000)
    assert set(result)=={'action'} and result['action'].shape==(50,14)
    assert np.isfinite(result['action']).all()
    actions.append(result['action'].copy());return actions[-1]
try:
    a=predict(obs);b=predict(obs)
    predict({**obs,'observation.state':np.full(14,0.5,dtype=np.float32)})
    assert not np.array_equal(a,actions[-1]),'state ignored'
    predict({**obs,'observation.images.cam_high':image('image',4)})
    assert not np.array_equal(a,actions[-1]),'camera ignored'
    api.reset(prompt='move the cup to the plate');predict(obs)
    assert not np.array_equal(a,actions[-1]),'prompt ignored'
    np.savez(out.with_suffix('.npz'),actions=np.stack(actions),latency_ms=latencies)
    result=dict(model_id=model,revision=revision,public_runtime=True,precision='native',
                action_shape=[50,14],parameter_dtypes=dtypes,setup_seconds=setup,
                repeat_maxabs=float(np.max(np.abs(a-b))),state_camera_prompt_changes=True,
                reset_verified=True,graph_stats=loop.graph_stats,latency_ms=latencies,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(inspect.getfile(type(loop))).read_bytes()).hexdigest(),
                scope='Native public API on recorded images and synthetic state; no new bit-exact or task-quality certificate')
    out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
finally:api.close()
