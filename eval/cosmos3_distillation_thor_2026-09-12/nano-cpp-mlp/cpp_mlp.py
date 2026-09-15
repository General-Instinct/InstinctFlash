"""Explicit offline BF16 NN projection runner; caller owns input and weights."""
import ctypes
from pathlib import Path
import torch
import torch.nn.functional as F

class CppMLP:
    def __init__(self, x, transposed_weights, library, *, tuned=False):
        self.lib=ctypes.CDLL(str(Path(library).resolve()))
        ptr=ctypes.c_void_p;integer=ctypes.c_int
        self.lib.ifl_error.restype=ctypes.c_char_p
        self.lib.ifl_create.restype=ptr
        self.lib.ifl_destroy.argtypes=[ptr]
        self.lib.ifl_run.argtypes=[ptr,ptr,ptr,ptr,integer,integer,integer,ptr]
        self.lib.ifl_tune.argtypes=[ptr,ptr,ptr,ptr,integer,integer,integer]
        self.handle=self.lib.ifl_create()
        if not self.handle:raise RuntimeError(self.lib.ifl_error().decode())
        self.x=x;self.weights=transposed_weights
        self.g=torch.empty((x.shape[0],12288),device=x.device,dtype=x.dtype)
        self.u=torch.empty_like(self.g);self.d=torch.empty_like(x)
        if tuned:
            assert torch.cuda.current_stream()==torch.cuda.default_stream()
            self.call('ifl_tune',x,self.weights[0],self.g)
            self.call('ifl_tune',x,self.weights[1],self.u)
            hidden=F.silu(self.g)*self.u
            self.call('ifl_tune',hidden,self.weights[2],self.d)
    def call(self,name,a,b,d):
        assert a.is_contiguous() and b.is_contiguous() and d.is_contiguous()
        assert a.dtype==b.dtype==d.dtype==torch.bfloat16
        assert a.device==b.device==d.device and a.is_cuda
        m,k=a.shape;n=b.shape[1]
        assert b.shape[0]==k and d.shape==(m,n)
        args=[self.handle,a.data_ptr(),b.data_ptr(),d.data_ptr(),m,n,k]
        if name=='ifl_run':args.append(torch.cuda.current_stream().cuda_stream)
        if getattr(self.lib,name)(*args):raise RuntimeError(self.lib.ifl_error().decode())
    def __call__(self):
        self.call('ifl_run',self.x,self.weights[0],self.g)
        self.call('ifl_run',self.x,self.weights[1],self.u)
        hidden=F.silu(self.g)*self.u
        self.call('ifl_run',hidden,self.weights[2],self.d)
        return self.d
    def close(self):
        torch.cuda.synchronize()
        if self.lib.ifl_destroy(self.handle):raise RuntimeError(self.lib.ifl_error().decode())
        self.handle=None
