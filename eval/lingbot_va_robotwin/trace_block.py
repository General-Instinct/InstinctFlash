#!/usr/bin/env python3
"""Dispatch-level trace of ONE LingBot-VA transformer block.

The profiler tells you what was slow. This tells you what actually *happened*: every aten op in
order, every tensor that got new storage versus a view, every dtype transition, every
data-dependent shape, and every allocation. That is the input a compiler needs; a kernel author
can get away with less, and that is exactly how you end up fusing the wrong thing.

Implemented with `TorchDispatchMode`, which sees every op after decomposition — including the ones
`torch.profiler` folds away and the ones that never appear in Python source (`aten::_to_copy` for
a `.float()`, the `expand`/`view` chain behind a broadcast, the `empty` behind every out-of-place
op).

For each op we record:
  * shapes and dtypes in and out
  * MATERIALIZATION: does the output own new storage, or is it a view of an input?
  * ROUNDING: does the output dtype have fewer mantissa bits than the compute dtype?
  * SYNC: is the output shape data-dependent (nonzero, item, tolist, local_scalar_dense)?
  * BYTES: allocated, read, written

Constructs a single block from the real config rather than loading the 10 GB checkpoint. The op
sequence and dtypes are a property of the code and the shapes, not of the weights.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

LINGBOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT, "wan_va"))
sys.path.insert(0, "/home/ubuntu/iwm_shims")

# real geometry, from transformer/config.json + va_robotwin_cfg.py
DIM, FFN, HEADS, EPS = 3072, 14336, 24, 1e-6
TEXT_LEN = 512
_MANTISSA = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24, torch.float64: 53}

_SYNC_OPS = {"nonzero", "_local_scalar_dense", "item", "masked_select", "unique", "unique2",
             "_unique2", "nonzero_numpy", "index_put_impl_"}
_VIEW_OPS = {"view", "_unsafe_view", "reshape", "permute", "transpose", "squeeze", "unsqueeze",
             "expand", "slice", "select", "detach", "t", "flatten", "unflatten", "alias",
             "as_strided", "split", "chunk", "narrow"}


@dataclass
class Ev:
    idx: int
    op: str
    in_shapes: list
    in_dtypes: list
    out_shape: tuple | None
    out_dtype: torch.dtype | None
    materialized: bool
    out_bytes: int
    rounding: str | None
    sync: bool
    kind: str


@dataclass
class Trace:
    events: list = field(default_factory=list)

    def add(self, e: Ev):
        self.events.append(e)


def _kind(op: str) -> str:
    if op in _VIEW_OPS:
        return "view"
    if op in _SYNC_OPS:
        return "sync"
    if any(s in op for s in ("mm", "matmul", "bmm", "linear", "addmm")):
        return "gemm"
    if any(s in op for s in ("layer_norm", "rms_norm", "softmax", "sum", "mean", "var")):
        return "reduction"
    if "attention" in op or "sdpa" in op or "scaled_dot" in op:
        return "attention"
    if op in ("copy_", "_to_copy", "to", "clone", "contiguous"):
        return "copy"
    if op in ("empty", "empty_like", "empty_strided", "zeros", "zeros_like", "full"):
        return "alloc"
    return "elementwise"


class BlockTracer(TorchDispatchMode):
    def __init__(self, trace: Trace):
        self.t = trace
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.split("::")[-1]
        ins = [a for a in args if isinstance(a, torch.Tensor)]
        in_ptrs = {a.untyped_storage().data_ptr() for a in ins}
        in_dtypes = [a.dtype for a in ins]
        widest_in = max((_MANTISSA.get(d, 0) for d in in_dtypes), default=0)

        out = func(*args, **kwargs)

        ot = out if isinstance(out, torch.Tensor) else (
            out[0] if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor)
            else None)
        materialized, ob, rounding, oshape, odtype = False, 0, None, None, None
        if ot is not None:
            odtype, oshape = ot.dtype, tuple(ot.shape)
            materialized = ot.untyped_storage().data_ptr() not in in_ptrs
            if materialized:
                ob = ot.untyped_storage().nbytes()
            om = _MANTISSA.get(odtype, 0)
            if widest_in and om and om < widest_in:
                rounding = f"{max(in_dtypes, key=lambda d: _MANTISSA.get(d,0))}->{odtype}"

        self.n += 1
        self.t.add(Ev(self.n, name, [tuple(a.shape) for a in ins], in_dtypes, oshape, odtype,
                      materialized, ob, rounding, name in _SYNC_OPS, _kind(name)))
        return out


def build_block(device, dtype):
    from modules.model import WanTransformerBlock
    blk = WanTransformerBlock(dim=DIM, ffn_dim=FFN, num_heads=HEADS,
                              cross_attn_norm=True, eps=EPS, attn_mode="torch")
    return blk.to(device=device, dtype=dtype).eval().requires_grad_(False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=240, help="240 = video stream, 32 = action")
    ap.add_argument("--batch", type=int, default=2, help="2 = CFG duplicated")
    ap.add_argument("--kv", type=int, default=9792, help="resident KV slots")
    ap.add_argument("--show", type=int, default=0, help="print the first N events")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    blk = build_block(dev, dt)
    blk.attn1.init_kv_cache("pos", args.kv, HEADS, DIM // HEADS, dev, dt, args.batch)

    B, N = args.batch, args.tokens
    h = torch.randn(B, N, DIM, device=dev, dtype=dt)
    enc = torch.randn(B, TEXT_LEN, DIM, device=dev, dtype=dt)
    temb = torch.randn(B, N, 6, DIM, device=dev, dtype=dt)
    rot = torch.randn(1, N, 1, DIM // HEADS // 2, device=dev, dtype=torch.complex64)

    with torch.no_grad():
        blk(h, enc, temb, rot, update_cache=0, cache_name="pos")   # warm
        tr = Trace()
        with BlockTracer(tr), torch.no_grad():
            blk(h, enc, temb, rot, update_cache=0, cache_name="pos")

    ev = tr.events
    print(f"=== ONE BLOCK: B={B} N={N} dim={DIM} ffn={FFN} heads={HEADS} kv_slots={args.kv} ===")
    print(f"total dispatched ops: {len(ev)}\n")

    by = collections.Counter(e.kind for e in ev)
    mat = [e for e in ev if e.materialized]
    print(f"{'kind':<14s} {'ops':>5s} {'materializing':>14s} {'MB allocated':>13s}")
    for k, c in by.most_common():
        m = [e for e in ev if e.kind == k and e.materialized]
        print(f"{k:<14s} {c:5d} {len(m):14d} {sum(x.out_bytes for x in m)/1e6:13.2f}")
    print(f"{'TOTAL':<14s} {len(ev):5d} {len(mat):14d} {sum(e.out_bytes for e in mat)/1e6:13.2f}")

    print(f"\n--- rounding points (dtype narrowing) ---")
    rp = [e for e in ev if e.rounding]
    rc = collections.Counter(f"{e.op} {e.rounding}" for e in rp)
    for k, c in rc.most_common(10):
        print(f"  {c:3d}x  {k}")
    print(f"  total: {len(rp)} narrowing ops in one block")

    print(f"\n--- synchronizations (data-dependent shape) ---")
    sy = [e for e in ev if e.sync]
    print(f"  {len(sy)}: {collections.Counter(e.op for e in sy).most_common()}")

    print(f"\n--- largest materializations ---")
    for e in sorted(mat, key=lambda x: -x.out_bytes)[:8]:
        print(f"  {e.out_bytes/1e6:8.2f} MB  {e.op:<20s} {e.out_shape} {e.out_dtype}")

    if args.show:
        print(f"\n--- first {args.show} ops in order ---")
        for e in ev[:args.show]:
            flag = "M" if e.materialized else "."
            r = f" [{e.rounding}]" if e.rounding else ""
            print(f"  {e.idx:4d} {flag} {e.kind:<12s} {e.op:<22s} {e.out_shape}{r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
