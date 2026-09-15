"""Current direct-checkpoint V2 staged chain: live BF16 vision, FP8 action expert."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoConfig
import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.vla2_thor import Vla2TorchFrontendThor
from flash_rt.models.vla2.moe_engine import Vla2MoeEngine
from instinctflash.runtime.vla2_engine import NativeVla2Vision,Vla2StagedEngine

out=Path(sys.argv[1])
if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('refusing overwrite')
home=Path.home()
checkpoint=home/'.cache/huggingface/hub/models--robbyant--lingbot-vla-v2-6b-robotwin/snapshots/0451855729ec904f970600e0aec8b84661423afe/checkpoints/global_step_50000/hf_ckpt'
base=home/'.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17'
vision=NativeVla2Vision(AutoConfig.from_pretrained(base).vision_config,checkpoint,'cuda')
moe=Vla2MoeEngine.from_checkpoint(fvk,checkpoint,mode='batched',schedule='batched',router_source='fp16')
frontend=Vla2TorchFrontendThor(checkpoint,use_cuda_graph=False,routed_moe_fn=moe.routed_moe_fn)
frontend.load_lm_fp16_stack(checkpoint)
frontend.lm_prefill_precision='fp16'
engine=Vla2StagedEngine(frontend,moe,vision)
dumps=[torch.load(home/f'thor_t2v2/p1_dumps/obs{i}.pt',map_location='cpu',weights_only=False) for i in (0,8)]
engine.set_prompt(dumps[0]['eng_ids'].tolist())
torch.manual_seed(817)
noise=torch.randn(50,55,dtype=torch.bfloat16)
actions=[];latencies=[]
for index in (0,1,0):
    d=dumps[index]
    result=engine.infer_staged(d['patches'],d['state55'],noise)
    actions.append(result['actions']);latencies.append(result['latency_ms'])
assert np.array_equal(actions[0],actions[2]),'fixed input did not restore after changed observation'
assert not np.array_equal(actions[0],actions[1]),'actions did not respond to observation'
# Isolate camera sensitivity from state changes.
result=engine.infer_staged(dumps[1]['patches'],dumps[0]['state55'],noise)
assert not np.array_equal(actions[0],result['actions']),'camera-insensitive action output'
actions.append(result['actions']);latencies.append(result['latency_ms'])
np.savez(out.with_suffix('.npz'),actions=np.stack(actions),latency_ms=latencies,
         expert_scales=frontend._exp_act_scales.cpu().numpy(),moe_scales=moe.dn_act.cpu().numpy())
result=dict(graph_replays=engine.replays,vision='native BF16 eager; three deepstack taps',
            prefill='FP16 from checkpoint',expert='FP8; FP16 router input',
            calibration='first real observation; maxima over all ten denoise steps; no historical artifacts',
            repeat_byte_equal=True,camera_changes_actions=True,latency_ms=latencies,
            artifacts_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
            source_sha256={c.__name__:hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in (Vla2StagedEngine,Vla2MoeEngine,Vla2TorchFrontendThor)},
            scope='Recorded patches, fixed BF16 noise, normalized 50x55 actions; not public Runtime or task-quality evidence')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
