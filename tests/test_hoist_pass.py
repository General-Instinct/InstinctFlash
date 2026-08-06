#!/usr/bin/env python3
"""HoistInvariant as a true engine pass: three cases, no model symbols in the pass.

  (a) LingBot-VA   real fp32 parameter casts, MODEL-scoped, evaluated at LAYER scope -> hoisted,
                   and the outputs must stay BIT-EXACT
  (b) synthetic    an episode-constant projection recomputed per step -> hoisted, on a structure
                   that is not a transformer and that no pass was designed around
  (c) synthetic    per-step noise, scope == evaluated_at -> correct NO-OP with a stated reason

    python tests/test_hoist_pass.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path += [os.path.join(LINGBOT_ROOT, "wan_va"), LINGBOT_ROOT,
             os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin")]

import torch

from instinctwm.passes.hoist_invariant import HoistInvariant
from instinctwm.passes.interface import Scope, SiteKind, run_pass

DEV, DT = torch.device("cuda"), torch.bfloat16
results = []


def pass_is_model_free() -> bool:
    import ast
    path = os.path.join(os.path.dirname(__file__), "..",
                        "instinctwm", "passes", "hoist_invariant.py")
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = getattr(node, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                node.body = b[1:]
    code = ast.unparse(tree)
    bad = [t for t in ("modules.model", "FP32LayerNorm", "scale_shift_table", "_iwm_w32",
                       "WanTransformer", ".blocks", "layer_norm", "cosmos") if t in code]
    print(f"  {'OK  ' if not bad else 'FAIL'} pass CODE references no model symbol"
          + (f" -- found {bad}" if bad else ""))
    return not bad


def case_a_lingbot() -> bool:
    print("\n=== (a) LingBot-VA: real fp32 parameter casts ===")
    import trace_block
    from trace_block import DIM, HEADS, TEXT_LEN
    from instinctwm.adapters.lingbot import LingBotSurface

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
    with torch.no_grad():
        ref = surf.stack(h, enc, tp, rot).clone()

    p = HoistInvariant()
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    with torch.no_grad():
        got = surf.stack(h, enc, tp, rot)
    nd = (got != ref).sum().item()
    print(f"  {'OK  ' if nd == 0 else 'FAIL'} BIT-EXACT after hoisting: differing={nd}  "
          f"({p.stats()})")
    if p.declines:
        print(f"  declines: {[str(d) for d in p.declines][:2]}")
    n_inv = p.invalidate(Scope.EPISODE)
    print(f"  invalidate(EPISODE) dropped {n_inv} cache(s) -- MODEL-scoped casts correctly SURVIVE")
    return res.fired and nd == 0 and n_inv == 0


def cases_bc_synthetic() -> bool:
    print("\n=== (b)+(c) synthetic adapter: one hoistable site, one that must not be ===")
    from instinctwm.adapters.synthetic import SyntheticSurface

    surf = SyntheticSurface(DEV)
    with torch.no_grad():
        ref = [surf.step().clone() for _ in range(3)]
    before = dict(surf.calls)

    p = HoistInvariant()
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    for d in p.declines:
        print(f"  DECLINE {d}")

    surf.calls.update(episode_projection=0, per_step_noise=0)
    with torch.no_grad():
        got = [surf.step() for _ in range(3)]
    after = dict(surf.calls)

    hoisted = res.applied == ("synthetic.episode_projection",)
    print(f"  {'OK  ' if hoisted else 'FAIL'} (b) only the EPISODE-scoped site was rewritten: "
          f"{list(res.applied)}")

    declined_right = any(d.site_id == "synthetic.per_step_noise" for d in p.declines)
    print(f"  {'OK  ' if declined_right else 'FAIL'} (c) the STEP-scoped site was declined "
          f"with a reason")

    # the hoisted producer runs once over 3 steps; the un-hoisted one still runs every step
    counts_ok = after["episode_projection"] == 1 and after["per_step_noise"] == 3
    print(f"  {'OK  ' if counts_ok else 'FAIL'} producer calls over 3 steps: "
          f"projection {before['episode_projection']}->{after['episode_projection']}, "
          f"noise {before['per_step_noise']}->{after['per_step_noise']}")

    same = all(torch.equal(a, b) for a, b in zip(ref, got))
    print(f"  {'OK  ' if same else 'FAIL'} outputs unchanged across all 3 steps (the per-step "
          f"value still varies, so hoisting it would have shown up here)")

    n_inv = p.invalidate(Scope.EPISODE)
    surf.calls["episode_projection"] = 0
    with torch.no_grad():
        surf.step()
    revived = surf.calls["episode_projection"] == 1
    print(f"  {'OK  ' if revived else 'FAIL'} invalidate(EPISODE) dropped {n_inv} cache; the "
          f"projection recomputed on the next step")

    return hoisted and declined_right and counts_ok and same and revived


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    print("=== 0. is the pass model-free? ===")
    results.append(pass_is_model_free())
    results.append(case_a_lingbot())
    results.append(cases_bc_synthetic())
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
