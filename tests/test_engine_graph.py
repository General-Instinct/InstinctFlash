#!/usr/bin/env python3
"""The Plan + Executor seam, proved on real LingBot blocks.

What is being tested is the ENGINE, not a model: the same Plan runs under the eager oracle and
under graph capture, the results are compared bit-for-bit, and both are benchmarked. The gate is
bit-exactness -- a graph that is fast and wrong is the failure mode this seam exists to prevent.

    python tests/test_engine_graph.py [n_blocks]
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT_ROOT, "wan_va"))
sys.path.insert(0, LINGBOT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin"))

import torch

from instinctwm.engine.executor import (
    CaptureFailed, EagerExecutor, GraphExecutor, report, verify_and_bench,
)
from instinctwm.engine.plan import BufferSpec, CaptureUnit, Plan, PlanBuffer
from trace_block import DIM, HEADS, TEXT_LEN, build_block

import modules.model as M

DEV, DT, KV, B = torch.device("cuda"), torch.bfloat16, 9792, 2
NBLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LIVE = 4096


def install_slice_addressing():
    """P003's addressing, which is what makes the model capturable at all.

    The stock path does `mask.nonzero()` per layer per forward: a data-dependent shape and a host
    round trip, either of which invalidates capture. Measured: stock raises
    cudaErrorStreamCaptureInvalidated; this does not.
    """
    def fwd(self, q, k, v, rotary_emb, update_cache=0, cache_name='pos'):
        kv = self.attn_caches[cache_name] if (self.attn_caches is not None) and (
            cache_name in self.attn_caches) else None
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            def rope(x, freqs):
                xo = torch.view_as_complex(x.to(torch.float64).reshape(
                    x.shape[0], x.shape[1], x.shape[2], -1, 2))
                return torch.view_as_real(xo * freqs).flatten(3).to(x.dtype)
            query = rope(query, rotary_emb)
            key = rope(key, rotary_emb)
        if kv is not None and kv['k'] is not None:
            n = key.shape[1]
            s = self._iwm_live
            kv['k'][:, s:s + n] = key
            kv['v'][:, s:s + n] = value
            key = kv['k'][:, :s + n]                 # slice: a view, static shape
            value = kv['v'][:, :s + n]
        hs = self.attn_op(query, key, value).flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](hs))
    M.WanAttention.forward = fwd


def build_plan(streams=((240, "video"), (32, "action"))):
    """A Plan with one capture unit per stream shape.

    The stack is the same 30-layer-shaped work per unit; only the token count differs, which is
    exactly the ShapeSpace for this model: 2 stream shapes, 1 KV bucket (padding measured at
    1.11-1.19x, cheaper than maintaining more graphs).
    """
    units, buffers, keep = [], [], {}
    for N, label in streams:
        blocks = []
        for _ in range(NBLOCK):
            b = build_block(DEV, DT)
            b.attn1.init_kv_cache("pos", KV, HEADS, DIM // HEADS, DEV, DT, B)
            b.attn1._iwm_live = LIVE
            blocks.append(b)
        keep[label] = blocks

        names = (f"hidden_{label}", f"encoder_{label}", f"temb_{label}", f"rot_{label}")
        buffers += [
            BufferSpec(names[0], (B, N, DIM), DT),
            BufferSpec(names[1], (B, TEXT_LEN, DIM), DT),
            BufferSpec(names[2], (B, N, 6, DIM), DT),
            BufferSpec(names[3], (1, N, 1, DIM // HEADS // 2), torch.complex64),
        ]

        def make_fn(bl):
            def fn(hidden, encoder, temb, rot):
                x = hidden
                for blk in bl:
                    x = blk(x, encoder, temb, rot, update_cache=0, cache_name="pos")
                return x
            return fn

        units.append(CaptureUnit(name="block_stack", fn=make_fn(blocks),
                                 inputs=names, output=f"out_{label}", shape_key=label))

    plan = Plan(model_id="lingbot-va-posttrain-robotwin", units=tuple(units),
                buffers=tuple(buffers),
                plan_buffer=PlanBuffer(fields=("kv_live", "step_index")),
                notes={"blocks_per_unit": str(NBLOCK),
                       "kv_policy": "padded to a single bucket"})
    return plan, keep


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    install_slice_addressing()
    plan, _keep = build_plan()
    print(plan.describe())

    inputs = {}
    for u in plan.units:
        g = torch.Generator(device="cpu").manual_seed(0)
        d = {}
        for n in u.inputs:
            spec = plan.buffer(n)
            if spec.dtype == torch.complex64:
                d[n] = torch.randn(spec.shape, generator=g, dtype=torch.complex64).to(DEV)
            else:
                d[n] = torch.randn(spec.shape, generator=g).to(DEV, spec.dtype)
        inputs[u.key] = d

    print("\n=== 1. capture ===")
    ex = GraphExecutor(plan, DEV)
    try:
        ex.prepare()
        print(f"  OK   captured {len(ex.graphs)} units: {list(ex.graphs)}")
    except CaptureFailed as e:
        print(f"  FAIL {e}")
        return 1

    print("\n=== 2. verify against the eager oracle + benchmark ===")
    res = verify_and_bench(plan, ex, DEV, lambda u: inputs[u.key])
    print(report(res))

    print("\n=== 3. KV growth: recapture must stay bit-exact across the episode ===")
    # The live set grows 272 tokens/cycle. Padding to capacity and masking is NOT bit-exact
    # (measured: 750/196608 elements differ), so the executor recaptures instead. This walks the
    # extent the way an episode does and checks every step against the oracle.
    oracle = EagerExecutor(plan, DEV)
    u = plan.units[1]                                     # action stream
    ins = inputs[u.key]
    grow_ok = True
    for cycle, live in enumerate(range(3264, 3264 + 272 * 6, 272)):
        for blk in _keep["action"]:
            blk.attn1._iwm_live = live
        ref = oracle.run(u.key, **ins).clone()
        got = ex.run(u.key, extent=live, **ins)
        nd = (got != ref).sum().item()
        grow_ok &= nd == 0
        print(f"  {'OK  ' if nd == 0 else 'FAIL'} cycle {cycle}  live={live:5d}  "
              f"differing={nd}  graphs held={len(ex.graphs)}  captures so far={ex.n_captures}")

    ok = (all(r.bit_exact for r in res) and all(r.speedup > 1.0 for r in res) and grow_ok)
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(r.bit_exact for r in res)}/{len(res)} bit-exact, "
          f"{sum(r.speedup > 1.0 for r in res)}/{len(res)} faster than eager, "
          f"growth {'clean' if grow_ok else 'BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
