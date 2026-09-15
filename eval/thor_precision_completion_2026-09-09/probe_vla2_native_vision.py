"""Verify current BF16 vision and deepstack features on two recorded V2 frames."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoConfig
from instinctflash.runtime.vla2_engine import NativeVla2Vision

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
home=Path.home()
checkpoint=home/'.cache/huggingface/hub/models--robbyant--lingbot-vla-v2-6b-robotwin/snapshots/0451855729ec904f970600e0aec8b84661423afe/checkpoints/global_step_50000/hf_ckpt'
base=home/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'
config=AutoConfig.from_pretrained(base).vision_config
vision=NativeVla2Vision(config,checkpoint,'cuda')
assert {p.dtype for p in vision.model.parameters()}=={torch.bfloat16}
results=[]
for index in (0,8,0):
    dump=torch.load(home/f'thor_t2v2/p1_dumps/obs{index}.pt',map_location='cpu',weights_only=False)
    merged,deepstack=vision(dump['patches'])
    results.append(torch.stack([merged,*deepstack]).float().cpu().numpy())
assert np.array_equal(results[0],results[2])
assert all(not np.array_equal(results[0][i],results[1][i]) for i in range(4))
np.savez(out.with_suffix('.npz'),features=np.stack(results))
result=dict(dtype='bfloat16',frames=[0,8,0],all_features_finite=True,
            repeat_byte_equal=True,all_four_features_change_with_camera=True,
            feature_shapes=[list(x.shape) for x in results],
            artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(inspect.getfile(NativeVla2Vision)).read_bytes()).hexdigest(),
            scope='Checkpoint BF16 vision, recorded patches; not full Runtime or closed-loop quality')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
