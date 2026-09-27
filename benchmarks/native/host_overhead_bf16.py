"""Host-side cost per call of the fused BF16 linear variants (GPU kept busy so launches never block)."""
import time

import torch

from instinctflash.kernels.gemm_bf16 import bf16_gemm_act_persistent


def host_us(fn, n=50):
    fn(); torch.cuda.synchronize()
    torch.cuda._sleep(400_000_000)   # keep the queue busy while timing the host path
    t = time.perf_counter()
    for _ in range(n):
        fn()
    host = (time.perf_counter() - t) / n * 1e6
    torch.cuda.synchronize()
    return host


def main():
    dev = "cuda"
    x = torch.randn(3093, 2048, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(9216, 2048, device=dev) * 0.02).to(torch.bfloat16)
    from instinctflash.backends import bf16_linear as B
    cands = {"triton persistent": lambda: bf16_gemm_act_persistent(x, w, config=B.THOR_CONFIG),
             "custom op (triton)": lambda: B._linear_relu2(x, w),
             "eager linear": lambda: torch.nn.functional.linear(x, w),
             "empty alloc": lambda: torch.empty(3093, 9216, device=dev, dtype=torch.bfloat16)}
    from triton.tools.tensor_descriptor import TensorDescriptor
    from instinctflash.kernels import gemm_bf16 as G
    out = torch.empty(3093, 9216, device=dev, dtype=torch.bfloat16)
    cands["3 descriptors only"] = lambda: (TensorDescriptor.from_tensor(x, [128, 64]),
                                           TensorDescriptor.from_tensor(w, [256, 64]),
                                           TensorDescriptor.from_tensor(out, [128, 128]))
    da, db, dd = (TensorDescriptor.from_tensor(x, [128, 64]), TensorDescriptor.from_tensor(w, [256, 64]),
                  TensorDescriptor.from_tensor(out, [128, 128]))
    sms = torch.cuda.get_device_properties(dev).multi_processor_count

    def launch_only():
        with G.compile_for(G.tc5_arch(x.device)):
            G._bf16_gemm_act_persistent[(sms,)](da, db, dd, 3093, 9216, 2048, BM=128, BN=256, BK=64, GROUP_M=8,
                                                ACT=1, NUM_SMS=sms, SPLIT_N=2, TC5=True, WS=True,
                                                num_warps=8, num_stages=4)
    cands["launch only (cached desc)"] = launch_only
    for name, fn in cands.items():
        print(f"{name:22s} host {host_us(fn):7.1f} us/call", flush=True)


if __name__ == "__main__":
    main()
