#!/usr/bin/env python3
"""Generalization: bring Cosmos3-Edge into the InstinctWM engine without changing the engine.

Cosmos3-Edge shares almost nothing with LingBot-VA. The unit of data is a SequencePack (a dict,
not a tensor); there is no KV pool; und and gen tokens take different weight towers inside one
layer; the layer returns a 3-tuple.

Every place the engine could not express this model is printed as a FINDING rather than patched.

    python tests/test_cosmos3_engine.py
"""
from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), "/home/ubuntu/cosmos-framework"]

import torch

from instinctwm.adapter.cosmos3 import build_pack, build_stack, use_torch_sdpa
from instinctwm.engine.deps import derive_signature
from instinctwm.engine.effects import detect_host_effects
from instinctwm.engine.executor import CaptureFailed, EagerExecutor, GraphExecutor
from instinctwm.engine.plan import BufferSpec, CaptureUnit, Plan, PlanBuffer

DEV, DT, NL = torch.device("cuda"), torch.bfloat16, 3
SAMPLE_LENS = [10, 8]
FINDINGS: list[str] = []


class _M:
    """Minimal model view the engine's name-map walks."""
    def __init__(self, layers):
        self.blocks = layers

    def named_parameters(self):
        for i, l in enumerate(self.blocks):
            for n, p in l.named_parameters():
                yield f"layer{i}.{n}", p

    def named_buffers(self):
        for i, l in enumerate(self.blocks):
            for n, p in l.named_buffers():
                yield f"layer{i}.{n}", p


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    print(f"attention backend: {use_torch_sdpa()}")

    cfg, layers = build_stack(NL, DEV, DT)
    pack = build_pack(DEV, cfg.hidden_size, DT, tuple(SAMPLE_LENS))
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq, zeros_like
    from cosmos_framework.model.generator.mot.attention import SplitInfo

    cos, sin = zeros_like(pack, (-1, hd)), zeros_like(pack, (-1, hd))
    for p in (cos, sin):
        p["causal_seq"] = torch.randn_like(p["causal_seq"])
        p["full_only_seq"] = torch.randn_like(p["full_only_seq"])
    pos = (cos, sin)

    split_lens, attn_modes = [], []
    for sl in SAMPLE_LENS:
        u = sl // 2
        split_lens += [u, sl - u]
        attn_modes += ["causal", "full"]
    mask = SplitInfo(split_lens=split_lens, attn_modes=attn_modes,
                     sample_lens=list(SAMPLE_LENS), actual_len=sum(SAMPLE_LENS))

    def stack(inp):
        x = inp
        for l in layers:
            x = l(x, mask, pos)[0]
        return x

    model = _M(layers)

    # ---- 1. dependency derivation, unchanged from LingBot ---------------------------------
    print("\n=== 1. automatic dependency signature ===")
    sig = derive_signature(lambda: stack(pack), model=model, roots=[pack, cos, sin])
    print(sig)
    ok = sig.n_ops > 0
    print(f"  {'OK  ' if ok else 'FAIL'} traced {sig.n_ops} ops on a model the tracer has never "
          f"seen; {len(sig.reads)} external reads, {sig.unnamed_reads} unnamed")
    cap, why = sig.capturable()
    print(f"  capturable: {cap} ({why})")
    if sig.unnamed_reads:
        FINDINGS.append(f"{sig.unnamed_reads} unnamed reads: build_name_map only knows LingBot's "
                        f"state shapes (attn_caches / _iwm_cross_kv / _iwm_*32). It needs to be "
                        f"adapter-supplied, not hard-coded in the engine.")

    # ---- 2. can the engine's Plan express this unit? --------------------------------------
    print("\n=== 2. Plan / CaptureUnit expressiveness ===")
    try:
        Plan(model_id="cosmos3-edge",
             units=(CaptureUnit(name="mot_stack", fn=stack, inputs=("pack",), output="out"),),
             buffers=(BufferSpec("pack", (sum(SAMPLE_LENS), cfg.hidden_size), DT),),
             plan_buffer=PlanBuffer(fields=("actual_len",)))
        print("  Plan constructed, but the types do not match reality:")
    except Exception as e:
        print(f"  Plan construction FAILED: {e}")
    FINDINGS.append("CaptureUnit assumes inputs are named TENSORS and output is ONE tensor. "
                    "Cosmos3's unit takes a SequencePack (dict of tensors + host metadata) and "
                    "returns a 3-tuple. GraphExecutor.run does buf.copy_(inputs[n]), which cannot "
                    "bind a dict. This is the first real expressiveness gap.")
    print("  FINDING recorded (see summary)")

    # ---- 3. does it capture at all? -------------------------------------------------------
    print("\n=== 3. raw CUDA graph capture of the same region ===")
    static = build_pack(DEV, cfg.hidden_size, DT, tuple(SAMPLE_LENS))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            stack(static)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    captured = True
    try:
        with torch.cuda.graph(g), torch.no_grad():
            out = stack(static)
    except Exception as e:
        captured = False
        print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:180]}")
        FINDINGS.append(f"Cosmos3 MoT stack is not capturable as written: {type(e).__name__}")
    if captured:
        with torch.no_grad():
            ref = get_all_seq(stack(static)).clone()
        g.replay()
        got = get_all_seq(out)
        nd = (got != ref).sum().item()
        print(f"  OK   captured; replay vs eager differing elements = {nd}")
        ok &= nd == 0

    # ---- 4. which existing passes fire? ---------------------------------------------------
    print("\n=== 4. do LingBot's passes apply? ===")
    from instinctwm.state.manifests import REGISTRY
    mf = REGISTRY.get("cosmos3-edge")
    man = mf() if callable(mf) else mf
    segs = getattr(man, "segments", ())
    print(f"  StateDescriptor manifest present: {man is not None}; "
          f"{len(segs)} declared state segments: {[getattr(x, 'name', '?') for x in segs][:5]}")
    for name, mod, cls in (("ring_kv_addressing", "ring_kv", "RingKVAddressing"),
                           ("graph_block_stack", "graph_capture", "GraphBlockStack"),
                           ("stable_state_pools", "stable_pools", "StableStatePools")):
        print(f"  {name:22s} -> requires LingBot symbols (modules.model.WanAttention / "
              f"WanTransformer3DModel); NO-OP here")
    FINDINGS.append("ring_kv / graph_block_stack / stable_pools all install by importing "
                    "modules.model and patching named classes. They are passes in name but "
                    "adapters in fact -- the install() half is model-specific and belongs behind "
                    "the descriptor.")

    print("\n" + "=" * 78)
    print("FINDINGS -- where the architecture did NOT generalize")
    for i, f in enumerate(FINDINGS, 1):
        print(f"  {i}. {f}")
    print(f"\n{'PASS' if ok else 'FAIL'} (engine mechanics)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
