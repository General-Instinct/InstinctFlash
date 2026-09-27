"""BF16 GEMM with a fused elementwise epilogue (Triton, TMA descriptors).

``D[M, N] = act(round_bf16(A[M, K] @ W[N, K]^T))`` with FP32 accumulation. The
epilogue first rounds the accumulator to BF16 (the eager linear's output
boundary) and then applies the activation in FP32 before the final BF16 store,
so ``relu2`` reproduces ``relu(linear(x)) ** 2`` evaluated as two BF16 ops.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from instinctflash.kernels._triton import compile_for, tc5_arch

_SMS = {}
_COMPILED = {}

RELU2 = 1
NONE = 0


@triton.jit
def _bf16_gemm_act_kernel(a_desc, b_desc, d_desc, M, N, K,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
                          ACT: tl.constexpr, STAGES: tl.constexpr, TC5: tl.constexpr, WS: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    acc = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, K, BK, num_stages=STAGES, warp_specialize=WS):
        a = a_desc.load([pid_m * BM, k])
        b = b_desc.load([pid_n * BN, k])
        acc = tl.dot(a, b.T, acc)
    y = acc.to(tl.bfloat16)
    if ACT == 1:
        f = tl.maximum(y.to(tl.float32), 0.0)
        y = (f * f).to(tl.bfloat16)
    d_desc.store([pid_m * BM, pid_n * BN], y)


DEFAULT_CONFIG = (128, 256, 64, 8, 2, 4)   # BM, BN, BK, GROUP_M, stages, warps[, ctas, warp_specialize]


def bf16_gemm_act(x, w, out=None, *, act=RELU2, config=None, tc5=True):
    """x [M, K] BF16 row-major, w [N, K] BF16 row-major -> [M, N] BF16."""
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    cfg = tuple(config or DEFAULT_CONFIG)
    BM, BN, BK, GM, stages, warps = cfg[:6]
    ctas = cfg[6] if len(cfg) > 6 else 1
    ws = cfg[7] if len(cfg) > 7 else False
    a_desc = TensorDescriptor.from_tensor(x, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(w, [BN, BK])
    d_desc = TensorDescriptor.from_tensor(out, [BM, BN])
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    arch = tc5_arch(x.device) if tc5 else None
    with compile_for(arch):
        _bf16_gemm_act_kernel[grid](a_desc, b_desc, d_desc, M, N, K, BM=BM, BN=BN, BK=BK, GROUP_M=GM,
                                    ACT=act, STAGES=stages, TC5=arch is not None, WS=ws, num_warps=warps,
                                    num_ctas=ctas)
    return out


@triton.jit
def _epilogue(acc, ACT: tl.constexpr):
    y = acc.to(tl.bfloat16)
    if ACT == 1:
        f = tl.maximum(y.to(tl.float32), 0.0)
        y = (f * f).to(tl.bfloat16)
    return y


@triton.jit
def _bf16_gemm_act_persistent(a_desc, b_desc, d_desc, M, N, K,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
                              ACT: tl.constexpr, NUM_SMS: tl.constexpr, SPLIT_N: tl.constexpr,
                              TC5: tl.constexpr, WS: tl.constexpr):
    """Persistent variant: one program per SM walks output tiles; the epilogue stores in
    ``SPLIT_N`` column slices so the TMA store overlaps the next tile's mainloop."""
    start = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    k_tiles = tl.cdiv(K, BK)
    tiles = num_m * num_n
    group = GROUP_M * num_n
    for tile in tl.range(start, tiles, NUM_SMS, flatten=True, warp_specialize=WS):
        gid = tile // group
        first_m = gid * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (tile % group) % size_m
        pid_n = (tile % group) // size_m
        acc = tl.zeros((BM, BN), tl.float32)
        for kk in range(k_tiles):
            a = a_desc.load([pid_m * BM, kk * BK])
            b = b_desc.load([pid_n * BN, kk * BK])
            acc = tl.dot(a, b.T, acc)
        if SPLIT_N == 2:
            a0, a1 = tl.split(tl.permute(tl.reshape(acc, (BM, 2, BN // 2)), (0, 2, 1)))
            d_desc.store([pid_m * BM, pid_n * BN], _epilogue(a0, ACT))
            d_desc.store([pid_m * BM, pid_n * BN + BN // 2], _epilogue(a1, ACT))
        else:
            d_desc.store([pid_m * BM, pid_n * BN], _epilogue(acc, ACT))


def bf16_gemm_act_persistent(x, w, out=None, *, act=RELU2, config=(128, 256, 64, 8, 4, 8, 2, True), tc5=True):
    """As :func:`bf16_gemm_act`; config = (BM, BN, BK, GROUP_M, stages, warps, split_n, warp_specialize).

    The first call for a (device, shape, config) compiles through Triton's JIT; later calls
    launch the retained compiled kernel directly. On Thor the JIT's per-call argument binding
    and specialization lookup cost more host time than the kernel's own launch, which matters
    in CPU-bound serving loops.
    """
    M, K = x.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    BM, BN, BK, GM, stages, warps, split, ws = config
    sms = _SMS.get(x.device)
    if sms is None:
        sms = _SMS[x.device] = torch.cuda.get_device_properties(x.device).multi_processor_count
    a_desc = TensorDescriptor(x, [M, K], [K, 1], [BM, BK])
    b_desc = TensorDescriptor(w, [N, K], [K, 1], [BN, BK])
    d_desc = TensorDescriptor(out, [M, N], [N, 1], [BM, BN // split])
    arch = tc5_arch(x.device) if tc5 else None
    args = (a_desc, b_desc, d_desc, M, N, K, BM, BN, BK, GM, act, sms, split, arch is not None, ws)
    key = (x.device, x.dtype, M, N, K, tuple(config), act, arch)
    kernel = _COMPILED.get(key)
    if kernel is None:
        with compile_for(arch):
            kernel = _bf16_gemm_act_persistent[(sms,)](*args, num_warps=warps, num_stages=stages)
        _COMPILED[key] = kernel
        return out
    stream = torch.cuda.current_stream(x.device).cuda_stream
    enter, leave = triton.knobs.runtime.launch_enter_hook, triton.knobs.runtime.launch_exit_hook
    meta = kernel.launch_metadata((sms, 1, 1), stream, *args) if enter is not None else None
    kernel.run(sms, 1, 1, stream, kernel.function, kernel.packed_metadata, meta, enter, leave, *args)
    return out
