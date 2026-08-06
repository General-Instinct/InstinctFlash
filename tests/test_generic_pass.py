#!/usr/bin/env python3
"""One pass, two structurally different WAMs, zero model imports in the pass.

`instinctwm/passes/graph_capture.py` contains no `import modules.model`, no class name, no
`for block in self.blocks`. It asks each adapter for CAPTURE_UNIT sites and rewrites them.

The comparison that matters is with `passes/lingbot/graph_capture.py`, the LingBot version,
which imports the model module, reaches for `WanTransformer3DModel`, rewrites the SOURCE of its
`forward`, and knows the argument names `update_cache` / `cache_name`.

    python tests/test_generic_pass.py
"""
from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), "/home/ubuntu/cosmos-framework"]
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path += [os.path.join(LINGBOT_ROOT, "wan_va"), LINGBOT_ROOT,
             os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin")]

import torch

from instinctwm.passes.graph_capture import GraphCapture
from instinctwm.passes.interface import SiteKind, run_pass

DEV, DT = torch.device("cuda"), torch.bfloat16
results = []


def source_is_model_free() -> bool:
    """Check the CODE, not the prose.

    The pass docstring deliberately names LingBot symbols in order to contrast itself with the
    old model-specific version, so a raw substring scan flags its own explanation. Strip
    docstrings and comments, then look at what the code actually references.
    """
    import ast

    path = os.path.join(os.path.dirname(__file__), "..",
                        "instinctwm", "passes", "graph_capture.py")
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):                        # drop every docstring
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    code = ast.unparse(tree)                           # comments are gone with the parse
    bad = [t for t in ("modules.model", "WanAttention", "WanTransformer3DModel", ".blocks",
                       "cosmos_framework", "SequencePack", "attn_caches", "update_cache",
                       "cache_name") if t in code]
    print(f"  {'OK  ' if not bad else 'FAIL'} the pass CODE references no model symbol"
          + (f" -- found {bad}" if bad else " (docstrings excluded; they cite the old pass)"))
    return not bad


def run_cosmos():
    print("\n=== Cosmos3-Edge (two-tower MoT, SequencePack) ===")
    from instinctwm.adapters.cosmos3 import (
        Cosmos3Surface, build_pack, build_stack, use_torch_sdpa,
    )
    from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq, zeros_like
    from cosmos_framework.model.generator.mot.attention import SplitInfo

    use_torch_sdpa()
    cfg, layers = build_stack(3, DEV, DT)
    SL = [10, 8]
    pack = build_pack(DEV, cfg.hidden_size, DT, tuple(SL))
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cos, sin = zeros_like(pack, (-1, hd)), zeros_like(pack, (-1, hd))
    for p in (cos, sin):
        p["causal_seq"] = torch.randn_like(p["causal_seq"])
        p["full_only_seq"] = torch.randn_like(p["full_only_seq"])
    sl, am = [], []
    for s in SL:
        u = s // 2
        sl += [u, s - u]
        am += ["causal", "full"]
    mask = SplitInfo(split_lens=sl, attn_modes=am, sample_lens=list(SL), actual_len=sum(SL))

    surf = Cosmos3Surface(layers, mask, (cos, sin))
    with torch.no_grad():
        ref = get_all_seq(surf.stack(pack)).clone()

    p = GraphCapture()
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    with torch.no_grad():
        got = get_all_seq(surf.stack(pack))
    nd = (got != ref).sum().item()
    print(f"  {'OK  ' if nd == 0 else 'FAIL'} after the pass: differing={nd}  {p.stats()}")
    results.append(res.fired and nd == 0)


def run_lingbot():
    print("\n=== LingBot-VA (dual-stream DiT, ring KV) ===")
    import trace_block
    from trace_block import DIM, HEADS, TEXT_LEN
    from instinctwm.adapters.lingbot import LingBotSurface
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing

    RingKVAddressing().install(None, type("S", (), {"_reset": lambda s, prompt=None: None}))
    B, N, KV, NL = 2, 32, 2048, 3
    blocks = []
    for _ in range(NL):
        b = trace_block.build_block(DEV, DT)
        b.attn1.init_kv_cache("pos", KV, HEADS, DIM // HEADS, DEV, DT, B)
        blocks.append(b)
    model = type("M", (), {"blocks": blocks})()
    h = torch.randn(B, N, DIM, device=DEV, dtype=DT)
    enc = torch.randn(B, TEXT_LEN, DIM, device=DEV, dtype=DT)
    tp = torch.randn(B, N, 6, DIM, device=DEV, dtype=DT)
    rot = torch.randn(1, N, 1, DIM // HEADS // 2, device=DEV, dtype=torch.complex64)

    surf = LingBotSurface(model)

    # 1. commit still inline -> the ADAPTER declares capturable=False, and the pass declines.
    p1 = GraphCapture()
    r1 = run_pass(p1, surf, DEV)
    print(f"  {r1}")
    declined = not r1.fired
    print(f"  {'OK  ' if declined else 'FAIL'} pass declines while the region mutates host state "
          f"(adapter said capturable=False)")

    # 2. defer the commit; the same pass now fires.
    type(blocks[0].attn1)._iwm_defer_commit = True
    with torch.no_grad():
        ref = surf.stack(h, enc, tp, rot).clone()
    p2 = GraphCapture()
    r2 = run_pass(p2, surf, DEV)
    print(f"  {r2}")
    with torch.no_grad():
        got = surf.stack(h, enc, tp, rot)
    nd = (got != ref).sum().item()
    print(f"  {'OK  ' if nd == 0 else 'FAIL'} after the pass: differing={nd}  {p2.stats()}")
    results.append(declined and r2.fired and nd == 0)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    print("=== 1. is the pass actually model-free? ===")
    results.append(source_is_model_free())
    run_cosmos()
    run_lingbot()
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
