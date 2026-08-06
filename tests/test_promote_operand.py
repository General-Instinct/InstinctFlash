#!/usr/bin/env python3
"""PromoteSmallOperand: widen the small operand, not the large one.

  (a) LingBot-VA   the modulation combine, BIT-EXACT, and the 35.4 MB activation cast is gone
  (b) decline      a value-preserving-widening violation (fp32 -> bf16 is NOT a widening)
  (c) decline      operands too close in size for the rewrite to pay for itself

    python tests/test_promote_operand.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path += [os.path.join(LINGBOT_ROOT, "wan_va"), LINGBOT_ROOT,
             os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin")]

import torch

from instinctwm.passes.interface import Site, SiteKind, run_pass
from instinctwm.passes.promote_small_operand import PromoteSmallOperand

DEV, DT = torch.device("cuda"), torch.bfloat16
results = []


def model_free() -> bool:
    import ast
    path = os.path.join(os.path.dirname(__file__), "..", "instinctwm", "passes",
                        "promote_small_operand.py")
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = getattr(n, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:]
    code = ast.unparse(tree)
    bad = [t for t in ("modules.model", "scale_shift_table", "WanTransformer", "temb",
                       ".blocks", "cosmos") if t in code]
    print(f"  {'OK  ' if not bad else 'FAIL'} pass CODE references no model symbol"
          + (f" -- found {bad}" if bad else ""))
    return not bad


def case_a() -> bool:
    print("\n=== (a) LingBot-VA modulation combine ===")
    import trace_block
    from trace_block import DIM, HEADS, TEXT_LEN
    from instinctwm.adapters.lingbot import LingBotSurface
    from instinctwm.passes.hoist_invariant import HoistInvariant

    B, N, KV, NL = 2, 240, 512, 3
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

    # hoist first: it caches the fp32 constant that the promotion rewrite then reuses
    hp = HoistInvariant()
    print(f"  {run_pass(hp, surf, DEV)}")
    pp = PromoteSmallOperand()
    res = run_pass(pp, surf, DEV)
    print(f"  {res}")
    print(f"  {pp.stats()}")

    with torch.no_grad():
        got = surf.stack(h, enc, tp, rot)
    nd = (got != ref).sum().item()
    d = (got.float() - ref.float()).abs().max().item()
    print(f"  {'OK  ' if nd == 0 else 'FAIL'} BIT-EXACT: differing={nd} max|d|={d:.3e}")
    return res.fired and nd == 0


class _Toy:
    """Two sites that must both be declined, for different reasons."""
    model_id = "toy"

    def sites(self, kind):
        if kind is not SiteKind.DTYPE_PROMOTION:
            return
        yield Site(kind=kind, id="toy.narrowing",
                   attrs={"narrow": torch.float32, "wide": torch.bfloat16,
                          "constant_elems": 10, "activation_elems": 100000,
                          "constant": lambda: torch.zeros(10)})
        yield Site(kind=kind, id="toy.same_size",
                   attrs={"narrow": torch.bfloat16, "wide": torch.float32,
                          "constant_elems": 1000, "activation_elems": 2000,
                          "constant": lambda: torch.zeros(1000)})

    def apply(self, rewrite):
        raise AssertionError("nothing should be applied")


def cases_bc() -> bool:
    print("\n=== (b)+(c) declines ===")
    p = PromoteSmallOperand()
    res = run_pass(p, _Toy(), DEV)
    print(f"  {res}")
    for d in p.declines:
        print(f"  DECLINE {d}")
    ids = {d.site_id for d in p.declines}
    good = (not res.fired) and ids == {"toy.narrowing", "toy.same_size"}
    print(f"  {'OK  ' if good else 'FAIL'} both declined: not-a-widening, and too-similar sizes")
    return good


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    print("=== 0. model-free? ===")
    results.append(model_free())
    results.append(case_a())
    results.append(cases_bc())
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
