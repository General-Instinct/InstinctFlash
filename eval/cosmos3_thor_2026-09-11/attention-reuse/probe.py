"""Thor dense non-causal BF16 FMHA screen; no model quality or BITEXACT certificate."""
import ctypes as c
import json
import math
from pathlib import Path
import sys
import torch
from natten.functional import attention

class FMHA:
    def __init__(self, library, max_q=4096, max_heads=64):
        assert torch.cuda.get_device_capability() == (11, 0)
        self.lib=c.CDLL(str(Path(library).resolve()))
        self.lib.bf16_fmha_create.argtypes=[c.c_int,c.c_int];self.lib.bf16_fmha_create.restype=c.c_void_p
        self.lib.bf16_fmha_destroy.argtypes=[c.c_void_p];self.lib.bf16_fmha_destroy.restype=None
        self.lib.bf16_fmha_run.argtypes=[c.c_void_p]*5+[c.c_int]*6+[c.c_float,c.c_void_p]
        self.lib.bf16_fmha_run.restype=c.c_int
        self.ctx=self.lib.bf16_fmha_create(max_q,max_heads)
        if not self.ctx:raise RuntimeError('FMHA allocation failed')
        self.max_q,self.max_heads=max_q,max_heads
        self.device=torch.cuda.current_device()
    def __call__(self,q,k,v,scale=None):
        if not (q.ndim==k.ndim==v.ndim==4 and q.shape[0]==k.shape[0]==v.shape[0]==1
                and q.is_cuda and q.device==k.device==v.device and q.device.index==self.device
                and q.dtype==k.dtype==v.dtype==torch.bfloat16 and k.shape==v.shape
                and q.shape[-1]==k.shape[-1]==128 and q.shape[2]%k.shape[2]==0
                and 0<q.shape[1]<=self.max_q and q.shape[2]<=self.max_heads
                and all(t.stride(-1)==1 and t.stride(-2)==128 for t in (q,k,v))
                and k.stride(1)==v.stride(1)):
            raise ValueError('Unsupported BF16 dense non-causal geometry')
        out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
        status=self.lib.bf16_fmha_run(self.ctx,q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),
            q.shape[1],k.shape[1],q.shape[2],k.shape[2],q.stride(1),k.stride(1),
            float(scale if scale is not None else 1/math.sqrt(128)),torch.cuda.current_stream(q.device).cuda_stream)
        if status:raise RuntimeError(f'BF16 FMHA failed: {status}')
        return out
    def close(self):
        if self.ctx:
            torch.cuda.synchronize(self.device);self.lib.bf16_fmha_destroy(self.ctx);self.ctx=None

def time_call(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(30):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)/30

def main():
    output=Path(sys.argv[1]);assert not output.exists()
    torch.manual_seed(707)
    torch.backends.cuda.matmul.allow_tf32=False
    op=FMHA(Path(__file__).with_name('bf16_fmha.so'))
    rows=[]
    try:
        for nq,nk,hq,hk in [(17,31,4,4),(129,257,4,4),(129,257,8,2),(3094,3182,32,32),(3094,3103,32,32)]:
            q=torch.randn(1,nq,hq,128,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(1,nk,hk,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
            y=op(q,k,v)
            # NATTEN requires matching heads on this installed backend.
            kk=k.repeat_interleave(hq//hk,dim=2);vv=v.repeat_interleave(hq//hk,dim=2)
            ref=attention(q,kk,vv,backend='cutlass-fmha')
            delta=(y.float()-ref.float()).abs()
            row={'shape':[nq,nk,hq,hk,128],'finite':bool(torch.isfinite(y).all()),
                'byte_equal':y.contiguous().view(torch.uint8).equal(ref.contiguous().view(torch.uint8)),
                'max_abs':float(delta.max()),'mae':float(delta.mean()),
                'candidate_ms':time_call(lambda:op(q,k,v)),
                'reference_ms':time_call(lambda:attention(q,kk,vv,backend='cutlass-fmha'))}
            if nq<256:
                gold=torch.nn.functional.scaled_dot_product_attention(q.float().transpose(1,2),kk.float().transpose(1,2),vv.float().transpose(1,2)).transpose(1,2)
                row['max_abs_vs_fp32']=float((y.float()-gold).abs().max())
            rows.append(row);print(json.dumps(row),flush=True)
        output.write_text(json.dumps({'torch':torch.__version__,'kind':'synthetic real-shape dense non-causal screen','rows':rows},indent=2)+'\n')
    finally:op.close()

if __name__=='__main__':main()
