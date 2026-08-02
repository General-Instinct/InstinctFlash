"""PromoteSmallOperand -- widen the small operand, not the large one.

Fourth engine pass, and the first that is not a port. It exists because the other half of P004 is
NOT hoisting and refused to fit in `HoistInvariant`:

    t = scale_shift_table[None] + temb.float()      # widens the 35.4 MB operand
    t = scale_shift_table32[None] + temb            # widens the 18 KB one; identical result

Nothing is being moved to a coarser scope here. The expression is being rewritten so the widening
lands on the cheap side, and type promotion inside the operator does the rest. That is a different
kind of transform, so it is a different pass -- folding it into the hoist would have made "hoist"
mean two unrelated things.

WHEN THIS IS BIT-EXACT

Only when the narrow dtype is a strict SUBSET of the wide one, so that promoting inside the
operator produces exactly the value an explicit cast would have. bf16 -> fp32 qualifies: every
bf16 is representable in fp32, and PyTorch's promotion is value-preserving. fp16 -> bf16 does not.
The pass checks the pair against a table rather than trusting a declaration, because getting this
wrong is silent.

WHEN IT IS WORTH IT

Only when the constant is meaningfully smaller than the activation. Rewriting a combine of two
equal-sized tensors buys nothing and still costs a wrapper, so the pass declines below a ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from instinctwm.passes.interface import Rewrite, RewriteKind, Site, SiteKind

#: narrow -> wide pairs where promotion inside an operator equals an explicit cast, exactly.
_EXACT_WIDENING = {
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float64),
    (torch.float16, torch.float64),
    (torch.float32, torch.float64),
}


@dataclass
class Decline:
    site_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.site_id}: {self.reason}"


class PromoteSmallOperand:
    name = "promote_small_operand"

    def __init__(self, min_ratio: float = 8.0, verbose: bool = False):
        #: how much bigger the activation must be than the constant to bother
        self.min_ratio = min_ratio
        self.verbose = verbose
        self.declines: list[Decline] = []
        self.rewritten: list[str] = []
        self.bytes_saved = 0

    def sites_required(self):
        return (SiteKind.DTYPE_PROMOTION,)

    # ---- decide -------------------------------------------------------------------------------
    def plan_rewrites(self, sites, device) -> list[Rewrite]:
        self.declines.clear()
        out: list[Rewrite] = []
        for site in sites.get(SiteKind.DTYPE_PROMOTION, []):
            why = self._why_not(site)
            if why:
                self.declines.append(Decline(site.id, why))
                if self.verbose:
                    print(f"[promote_small_operand] DECLINE {site.id}: {why}", flush=True)
                continue
            a = site.attrs
            saved = a["activation_elems"] * torch.empty(0, dtype=a["wide"]).element_size()
            self.bytes_saved += saved
            self.rewritten.append(site.id)
            out.append(Rewrite(
                site_id=site.id, kind=RewriteKind.WRAP, payload=self._rewrite(site),
                note=(f"widen the {a['constant_elems']}-element constant instead of the "
                      f"{a['activation_elems']}-element activation; saves {saved/1e6:.1f} MB "
                      f"per evaluation")))
        return out

    def _why_not(self, site: Site) -> str | None:
        a = site.attrs
        narrow, wide = a.get("narrow"), a.get("wide")
        if narrow is None or wide is None:
            return "site does not declare its narrow/wide dtypes"
        if (narrow, wide) not in _EXACT_WIDENING:
            return (f"{narrow} -> {wide} is not a value-preserving widening, so promoting inside "
                    f"the operator would not equal the explicit cast")
        ce, ae = a.get("constant_elems"), a.get("activation_elems")
        if not ce or not ae:
            return "site does not declare operand sizes, so the win cannot be estimated"
        if ae / ce < self.min_ratio:
            return (f"activation is only {ae/ce:.1f}x the constant (threshold {self.min_ratio}x); "
                    f"the rewrite would not pay for its own wrapper")
        if a.get("constant") is None:
            return "site exposes no constant accessor"
        return None

    # ---- what to install ----------------------------------------------------------------------
    def _rewrite(self, site: Site):
        constant = site.attrs["constant"]
        wide = site.attrs["wide"]

        def wrap(_orig_combine):
            def combine(activation):
                c = constant()
                if c.dtype is not wide:
                    c = c.to(wide)          # cheap: this is the SMALL operand
                # No `.to(wide)` on the activation. Promotion happens inside the add, which for a
                # value-preserving widening yields exactly the explicitly-cast result.
                return c + activation
            return combine

        return wrap

    def stats(self) -> str:
        return (f"rewritten={len(self.rewritten)} declined={len(self.declines)} "
                f"bytes_saved_per_eval={self.bytes_saved/1e6:.1f}MB")
