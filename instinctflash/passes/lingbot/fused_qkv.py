"""Fused Q/K/V projection, BITEXACT under a per-shape certificate. IMPLEMENTED, GATED, **NOT SHIPPED**.

VERDICT FIRST. Every correctness gate passes and the cycle gate does not:

    certificate   production shapes M=64 and M=480 certify exact (0 of 589,824 and 0 of 4,423,680
                  words differ). M=7, injected deliberately, REFUSES: 55 of 64,512 words differ, and
                  the pass falls back to the split projections for it.
    action gate   max|delta action| = 0 over 6 paired seeded cycles, with 1,798 fused calls -- so the
                  exactness is not the exactness of having done nothing.
    CYCLE GATE    ABBA: baseline 334.6/341.7 -> 338.15 ms, fused 340.2/337.3 -> 338.75 ms.
                  0.2% SLOWER, against 2.1% drift on the repeated base arm. The predicted 1.9% does
                  not appear. `passes/contract.py` rejects a pass that does not improve its declared
                  cost term regardless of tier, so this does not ship.

WHY THE REGION ESTIMATE DID NOT TRANSLATE, third time in a row. Isolated, the fused GEMM is 2.08x at 32
tokens and 1.18x at 240, which extrapolated to 6.13 ms/cycle. In situ it is worth nothing. The three
split GEMMs evidently already overlap with surrounding work, so serialising them into one larger GEMM
shortens no critical path -- a region benchmark measures a kernel in isolation and the cycle does not run
it in isolation. RoPE (1.10x region -> 0.3% cycle), the cast hoist (1.4% predicted -> 0.66% unresolvable)
and this (1.9% predicted -> 0.2% slower) now make the same point three times.

WHAT IS WORTH KEEPING. The certificate mechanism, and one hard fact it produced: **the tile_k invariant is
NOT universal.** M=7 breaks it on this exact stack, today. LAYER5_QKV_EXACTNESS.md argued that fused-QKV
exactness is a theorem conditional on a cuBLAS heuristic rather than a theorem outright; M=7 is the
counterexample that settles the argument. Anyone tempted to fuse GEMMs and claim bit-exactness from three
sampled shapes should read that line: three shapes passing does not make a fourth pass.

THE MECHANISM, for whoever needs it next:
  * shapes are DERIVED from descriptors (`frame_chunk_size`, `patch_size`, `latent_*`, `action_per_frame`),
    never listed, so a new operating point yields new shapes and certifies them on their own merits
  * certification is per shape, on integer bit patterns, not on a tolerance
  * latent geometry is assigned at the first `_reset()`, so certification defers to first sight of each
    shape -- the first call on any unknown shape always takes the split path, and the decision is only
    trusted from the second call onwards
  * there is no NUMERIC path. The only two states are "certified and fused" and "not certified, unchanged"

WHY THIS IS BITEXACT AND NOT NUMERIC. For `C[m,n] = sum_k A[m,k]*B[k,n]`, concatenating B along N adds
columns; every output element reduces over K, and K is untouched. N is embarrassingly parallel. So fused
is bit-identical to split whenever the K-loop is performed identically -- and measured on this stack it
is: `tile_k = 64` in all six selected kernels, both forms deterministic, 0 differing words at all three
production shapes. See LAYER5_QKV_EXACTNESS.md.

BUT tile_k STABILITY IS A cuBLAS HEURISTIC, NOT A CONTRACT. Nothing in the API promises it under a change
of N, and it is free to differ by library version, driver, architecture, or available workspace. So the
pass does not TRUST the invariant, it CERTIFIES it:

    at install, for every shape in the DECLARED envelope, compute split and fused and compare bit
    patterns. Fuse only where they match exactly. Anything else takes the original split path.

A backend or software change therefore costs performance, never correctness. If a future cuBLAS splits K
at N=9216, certification fails for that shape, the pass reports it, and the split projections run.

THERE IS NO NUMERIC FALLBACK, deliberately. A "close enough" path would mean this pass could silently
degrade the chain's bit-exactness claim after a library upgrade, with nothing in the log to say so. The
only two states are "certified and fused" and "not certified, unchanged".

FAIL-CLOSED AT RUNTIME TOO, not only at install. The declared envelope is derived from descriptors, but a
shape can still arrive that was never certified -- a new operating point, an unexpected token count, a
code path nobody enumerated. The forward path looks the shape up and falls back on a miss, so an
un-enumerated shape is slow rather than wrong.

HOW IT COMPOSES WITH P003. `ring_kv` is frozen and calls `self.to_q(q), self.to_k(k), self.to_v(v)`
directly. A frozen pass may only change for a correctness bug, so this fuses UNDERNEATH it: `to_q`
computes the fused GEMM and hands back its own slice while stashing the other two, and `to_k`/`to_v`
consume the stash. The stash is single-use and keyed on tensor identity, so any call pattern other than
the expected three-in-a-row misses it and falls back.

STATUS: NEGATIVE RESULT
1.9% predicted, 0.2% SLOWER measured. Its own per-shape certificate disproved the
invariant it was built on: M=7 differs in 55 of 64,512 words, so cuBLAS tile_k invariance is not universal
across the served envelope.
See HISTORICAL.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from instinctflash.passes.contract import (
    Applicability,
    BenchResult,
    CostTerm,
    Discovery,
    HardwareReq,
    Tier,
    VerifyResult,
)


@dataclass
class ShapeKey:
    """A GEMM shape from the declared envelope. Hashable, and printable in a log line."""

    m: int
    k: int
    n_split: int

    def key(self) -> tuple[int, int, int]:
        return (self.m, self.k, self.n_split)

    def __str__(self) -> str:
        return f"M={self.m} K={self.k} N={self.n_split}->{self.n_split * 3}"


@dataclass
class _Report:
    certified: dict[tuple[int, int, int], bool] = field(default_factory=dict)
    detail: dict[tuple[int, int, int], str] = field(default_factory=dict)
    fused_calls: int = 0
    fallback_calls: int = 0
    uncertified_shapes: set = field(default_factory=set)
    lazily_certified: set = field(default_factory=set)


def shape_envelope(server, spec=None) -> list[ShapeKey]:
    """Derive every production GEMM shape from DESCRIPTORS, not from the three we happened to measure.

    The shapes that occur today (M=64 action, M=480 video/kv_refresh) are OUTPUTS of this function. A
    different operating point declares a different geometry and yields different shapes, which are then
    certified on their own merits -- that is the point of deriving rather than listing.

    The token geometry comes from the server's own declared config, using the same formula the server
    uses (wan_va_server.py:398-402):

        latent tokens/chunk = frame_chunk_size * latent_height * latent_width / prod(patch_size)
        action tokens/chunk = frame_chunk_size * action_per_frame
        batch               = 2 when CFG is on, else 1

    RAISES rather than returning an empty list when the geometry cannot be read. The first version
    caught every exception and returned [], which made the pass a silent no-op: it "passed" the
    bit-exactness gate by doing nothing, and only the gate's own "the fused path actually ran" check
    caught it. A derivation that cannot derive must say so.
    """
    at = server.transformer.blocks[0].attn1
    k_dim, n_split = int(at.to_q.weight.shape[1]), int(at.to_q.weight.shape[0])
    jc = getattr(server, "job_config", None)
    if jc is None:
        raise RuntimeError("fused_qkv: server exposes no job_config; cannot derive the shape envelope")

    missing = [n for n in ("frame_chunk_size", "patch_size", "action_per_frame")
               if getattr(jc, n, None) is None]
    # `latent_height`/`latent_width` are assigned in `_reset()` (wan_va_server.py:391-396), not in
    # __init__, so at install time -- which is before the first reset -- they do not exist. That is not
    # a failure to derive, it is a derivation that has to wait, and the lazy path below covers it.
    for n in ("latent_height", "latent_width"):
        if getattr(server, n, None) is None:
            missing.append(n)
    if missing:
        raise RuntimeError(f"cannot derive the shape envelope yet; missing {missing} (latent geometry "
                           f"is set at the first reset). Shapes will be certified on first sight "
                           f"instead, which is equally fail-closed.")

    fcs = int(jc.frame_chunk_size)
    ps = jc.patch_size
    pprod = int(ps[0]) * int(ps[1]) * int(ps[2])
    latent_tokens = (fcs * int(server.latent_height) * int(server.latent_width)) // pprod
    action_tokens = fcs * int(jc.action_per_frame)
    batch = 2 if getattr(server, "use_cfg", False) else 1

    tokens = {latent_tokens, action_tokens}
    # A declared phase may carry its own token count; prefer it where present, since it is a
    # checkpoint fact rather than a geometry re-derivation.
    if spec is not None:
        for ph in getattr(spec, "phases", ()) or ():
            n = getattr(ph, "tokens", None)
            if n:
                tokens.add(int(n))
    tokens.discard(0)
    if not tokens:
        raise RuntimeError("fused_qkv: derived an empty token set; refusing to install")
    return [ShapeKey(m=batch * t, k=k_dim, n_split=n_split) for t in sorted(tokens)]


def _certify_shape(at, fused_w, fused_b, shp: ShapeKey) -> tuple[bool, str]:
    """Bit-compare split vs fused at one shape. Returns (exact, detail).

    Compared on the INTEGER BIT PATTERNS, not with a tolerance: the claim is exactness, and `==` on
    floats would accept -0.0 for 0.0 and would say nothing about NaN payloads.
    """
    w, dt, dev = at.to_q.weight, at.to_q.weight.dtype, at.to_q.weight.device
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        # Adversarial-ish magnitudes: values that make the K-reduction cancel are where a different
        # accumulation order would first show up.
        x = (torch.randn(shp.m, shp.k, generator=g) * 0.05 + 1.0).to(device=dev, dtype=dt)
        qs = torch.nn.functional.linear(x, at.to_q.weight, at.to_q.bias)
        ks = torch.nn.functional.linear(x, at.to_k.weight, at.to_k.bias)
        vs = torch.nn.functional.linear(x, at.to_v.weight, at.to_v.bias)
        y = torch.nn.functional.linear(x, fused_w, fused_b)
        qf, kf, vf = y.split([shp.n_split] * 3, dim=-1)
        ndiff = 0
        for a_, b_ in ((qs, qf), (ks, kf), (vs, vf)):
            a_i = a_.contiguous().view(torch.int16)
            b_i = b_.contiguous().view(torch.int16)
            ndiff += int((a_i != b_i).sum())
        total = qs.numel() * 3
        if ndiff == 0:
            return True, f"0/{total} words differ"
        return False, f"{ndiff}/{total} words differ -- NOT certified, split path retained"
    except Exception as e:
        return False, f"certification raised {type(e).__name__}: {e} -- split path retained"


class FusedQKVProjection:
    """One GEMM for Q/K/V where certified; the original three where not."""

    name = "fused_qkv_projection"
    hardware = HardwareReq()
    cost_term = CostTerm.PER_STEP

    def __init__(self) -> None:
        self.report = _Report()

    def applicability(self, spec, device) -> Applicability:
        return Applicability(
            True,
            "self-attention projects one shared input three times; concatenating the weights along "
            "out_features gives one GEMM. BITEXACT only for shapes where split and fused bit-compare "
            "equal at install; every other shape keeps the split path.",
            discovery=Discovery.DECLARED, cost_term=CostTerm.PER_STEP, claimed_tier=Tier.BITEXACT)

    def expected_delta_ms(self, spec, device) -> float:
        return 6.13      # measured region deltas x 30 blocks x forwards; see LAYER5_QKV_FEASIBILITY.md

    # ---- install ------------------------------------------------------------------------------
    def install(self, server_module, server, spec=None) -> list[str]:
        rep = self.report
        blocks = list(server.transformer.blocks)
        if not blocks:
            return []
        at0 = blocks[0].attn1
        if getattr(at0, "_iwm_fused_qkv", False):
            return []

        envelope: list[ShapeKey] = []
        try:
            envelope = shape_envelope(server, spec)
            print(f"InstinctFlash fused-qkv: envelope derived from descriptors -> "
                  f"{len(envelope)} shape(s): {', '.join(str(s) for s in envelope)}", flush=True)
        except RuntimeError as e:
            # NOT a failure to install. Certification is deferred to first sight of each shape, which
            # is equally fail-closed: the first call on an unknown shape always takes the split path.
            print(f"InstinctFlash fused-qkv: {e}", flush=True)

        # Certify once against block 0's weights. The GEMM shape and the kernel selection depend on
        # (M, N, K) and dtype, not on the weight VALUES, so one certification per shape is sound --
        # and it is verified per block anyway by the action-level gate.
        fw0 = torch.cat([at0.to_q.weight, at0.to_k.weight, at0.to_v.weight], dim=0).contiguous()
        fb0 = (torch.cat([at0.to_q.bias, at0.to_k.bias, at0.to_v.bias], dim=0).contiguous()
               if at0.to_q.bias is not None else None)
        for shp in envelope:
            ok, detail = _certify_shape(at0, fw0, fb0, shp)
            rep.certified[shp.key()] = ok
            rep.detail[shp.key()] = detail
            print(f"InstinctFlash fused-qkv: {'CERTIFIED ' if ok else 'REFUSED   '}{shp}   {detail}",
                  flush=True)

        n_ok = sum(1 for v in rep.certified.values() if v)
        if envelope and n_ok == 0:
            # A derived envelope that certifies nothing is a real refusal: the invariant does not hold
            # on this stack, so the pass declines rather than fusing anything.
            print("InstinctFlash fused-qkv: NO shape in the derived envelope certified; declining to "
                  "install. The split projections are unchanged.", flush=True)
            return []

        # ---- install the fused path, per attention module -------------------------------------
        for blk in blocks:
            at = blk.attn1
            fw = torch.cat([at.to_q.weight, at.to_k.weight, at.to_v.weight], dim=0).contiguous()
            fb = (torch.cat([at.to_q.bias, at.to_k.bias, at.to_v.bias], dim=0).contiguous()
                  if at.to_q.bias is not None else None)
            n = at.to_q.weight.shape[0]
            k_dim = at.to_q.weight.shape[1]
            state = {"pending": None}

            oq, ok_, ov = at.to_q.forward, at.to_k.forward, at.to_v.forward

            def q_fwd(x, _oq=oq, _fw=fw, _fb=fb, _n=n, _k=k_dim, _st=state, _at=at):
                _st["pending"] = None
                m = 1
                for d in x.shape[:-1]:
                    m *= int(d)
                sk = (m, _k, _n)
                if sk not in rep.certified:
                    # FIRST SIGHT of a shape the envelope did not name. Certify it now, but serve THIS
                    # call from the split path regardless: the decision is only trusted from the next
                    # call onwards. That is what makes an un-enumerated shape slow rather than wrong.
                    ok, detail = _certify_shape(_at, _fw, _fb, ShapeKey(m, _k, _n))
                    rep.certified[sk] = ok
                    rep.detail[sk] = detail
                    rep.lazily_certified.add(sk)
                    print(f"InstinctFlash fused-qkv: {'CERTIFIED ' if ok else 'REFUSED   '}"
                          f"{ShapeKey(m, _k, _n)}   {detail}   [first sight at runtime]", flush=True)
                    rep.fallback_calls += 1
                    return _oq(x)
                if rep.certified[sk] is not True:
                    rep.fallback_calls += 1
                    return _oq(x)
                y = torch.nn.functional.linear(x, _fw, _fb)
                qo, ko, vo = y.split([_n, _n, _n], dim=-1)
                _st["pending"] = (x, ko, vo)
                rep.fused_calls += 1
                return qo

            def k_fwd(x, _ok=ok_, _st=state):
                p = _st["pending"]
                # Identity check, not equality: the stash is only valid for the exact tensor `to_q`
                # was called with, and it is single-use. Any other pattern falls back.
                if p is not None and p[0] is x:
                    return p[1]
                return _ok(x)

            def v_fwd(x, _ov=ov, _st=state):
                p = _st["pending"]
                if p is not None and p[0] is x:
                    _st["pending"] = None          # consume: the stash never outlives its forward
                    return p[2]
                return _ov(x)

            at.to_q.forward, at.to_k.forward, at.to_v.forward = q_fwd, k_fwd, v_fwd
            at._iwm_fused_qkv = True
            at._iwm_fused_w, at._iwm_fused_b = fw, fb      # keep alive

        print(f"InstinctFlash fused-qkv: fusing {n_ok} of {len(envelope)} shapes across "
              f"{len(blocks)} blocks", flush=True)
        return [self.name]

    def stats(self) -> dict:
        r = self.report
        return {"fused_calls": r.fused_calls, "fallback_calls": r.fallback_calls,
                "certified": {f"M={k[0]}": v for k, v in r.certified.items()},
                "certified_at_install": sorted(k for k in r.certified
                                               if k not in r.lazily_certified),
                "certified_on_first_sight": sorted(r.lazily_certified),
                "uncertified_seen_at_runtime": sorted(
                    k for k, v in r.certified.items() if v is not True)}

    # ---- gates --------------------------------------------------------------------------------
    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(
            passed=(d == 0.0),
            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
            max_abs_delta=d,
            detail="fusion is only installed for shapes that bit-compared equal at install, so a "
                   "nonzero delta means a shape reached the fused path without being certified")

    def benchmark(self, harness) -> BenchResult:
        return harness.latency_ab(self.name)
