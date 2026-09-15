"""Original Nano generation MLP weights, synthetic profiled-size inputs: cost screen only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from instinctflash.backends.bf16_pointwise import silu_table, swiglu

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--library',type=Path,required=True)
a=p.parse_args()
if a.output.exists():p.error('Use a fresh output')
report=dict(category='SCREEN',synthetic_inputs=True,task_quality_certified=False,
    end_to_end_benchmark=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=[])
with open('/tmp/thor_gpu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert torch.cuda.get_device_capability()==(11,0)
    others=subprocess.check_output(['/usr/sbin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    assert not [pid for pid in others if int(pid)!=os.getpid()], 'GPU contention'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(271828)
    report.update(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    index_path=a.checkpoint/'transformer/diffusion_pytorch_model.safetensors.index.json'
    index=json.loads(index_path.read_text())['weight_map']
    report['cpp_sources']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('cpp_mlp.py','shim.cu','gemm_runner.cu','gemm_runner.h')}
    report['library_sha256']=hashlib.sha256(a.library.read_bytes()).hexdigest()
    report['index_sha256']=hashlib.sha256(index_path.read_bytes()).hexdigest()
    with torch.inference_mode():
        lut=silu_table(torch.device('cuda'))
        for layer in (0,35):
            weights={};identities={}
            for projection in ('gate','up','down'):
                key=f'layers.{layer}.mlp_moe_gen.{projection}_proj.weight'
                with safe_open(a.checkpoint/'transformer'/index[key],framework='pt',device='cpu') as file:
                    weight=file.get_tensor(key)
                assert weight.dtype==torch.bfloat16
                identities[key]=dict(shape=list(weight.shape),sha256=hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),shard=index[key])
                weights[projection]=weight.cuda()
            gate,up,down=(weights[n] for n in ('gate','up','down'))
            assert gate.shape==up.shape==(12288,4096) and down.shape==(4096,12288)
            packed=torch.cat((gate,up),dim=0)
            x=torch.randn((3093,4096),device='cuda',dtype=torch.bfloat16)
            def separate():return F.linear(F.silu(F.linear(x,gate))*F.linear(x,up),down)
            def separate_shared():return F.linear(swiglu(F.linear(x,gate),F.linear(x,up),lut),down)
            def merged():
                g,u=F.linear(x,packed).chunk(2,dim=-1)
                return F.linear(F.silu(g)*u,down)
            def merged_shared():
                g,u=F.linear(x,packed).chunk(2,dim=-1)
                return F.linear(swiglu(g.contiguous(),u.contiguous(),lut),down)
            from cpp_mlp import CppMLP
            transposed=[w.t().contiguous() for w in (gate,up,down)]
            default=CppMLP(x,transposed,a.library)
            tuned=CppMLP(x,transposed,a.library,tuned=True)
            variants=dict(separate=separate,separate_shared=separate_shared,cpp_default=default,cpp_tuned=tuned)
            reference=separate();assert torch.isfinite(reference).all()
            rows={}
            for name,fn in variants.items():
                output=fn();assert torch.isfinite(output).all()
                delta=(output.float()-reference.float()).abs()
                rows[name]=dict(maxabs=delta.max().item(),meanabs=delta.mean().item(),byte_identical=torch.equal(output.view(torch.uint8),reference.view(torch.uint8)),trials_ms=[])
                for _ in range(5):fn()
            torch.cuda.synchronize()
            names=list(variants)
            for trial in range(5):
                for name in names[trial%4:]+names[:trial%4]:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):variants[name]()
                    end.record();end.synchronize()
                    rows[name]['trials_ms'].append(begin.elapsed_time(end)/20)
            for row in rows.values():row['median_ms']=statistics.median(row['trials_ms'])
            report['results'].append(dict(layer=layer,input_shape=list(x.shape),weights=identities,variants=rows))
            print(json.dumps(dict(layer=layer,variants=rows)),flush=True)
            default.close();tuned.close()
    report['ok']=True
    a.output.write_text(json.dumps(report,indent=2)+'\n')
