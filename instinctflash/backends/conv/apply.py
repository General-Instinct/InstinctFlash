"""Apply a `ConvPlan` to a live module tree. The only part of this layer that touches a model.

SEPARATE FROM SELECTION ON PURPOSE. `registry.select()` decides and returns a plan; this applies one.
Keeping them apart is what lets the decision be inspected -- and refused -- before anything is mutated,
and it is the same decide/act seam the rest of the stack uses (ARCHITECTURE.md, Seam 2).

WHAT APPLYING MEANS. Converting the CONV WEIGHTS is sufficient: PyTorch propagates the memory format
from weights to activations through the convolution, so the encoder's intermediates follow without
being touched individually. That is also why the conversion amortises -- one decision per subgraph, not
one copy per operator.

`module.to(memory_format=...)` cannot be used on the whole tree: it applies to every parameter and
raises "required rank 5 tensor to use channels_last_3d format" on the rank-1 RMSNorm weights. Only the
Conv3d weights have a 5-D layout to change.

EVERY CONV SUBGRAPH MUST BE CONVERTED, NOT JUST THE OBVIOUS ONE. LingBot-VA runs two VAEs -- a
full-resolution one for the head camera and a half-resolution one for the two wrist cameras. Converting
only the first leaves two thirds of the observation encode on the fallback path, which would show up as
a disappointing end-to-end number and be misread as the optimization not working.
"""

from __future__ import annotations

import torch

from instinctflash.autotune import Candidate, Decision, Site, autotune, register_site
from instinctflash.backends.conv.registry import ConvPlan
from instinctflash.backends.conv.semantics import MemoryLayout
from instinctflash.passes.contract import Tier


def convertible_convs(module) -> list[torch.nn.Conv3d]:
    """Conv3d modules whose weight has a 5-D layout to change."""
    return [m for m in module.modules()
            if isinstance(m, torch.nn.Conv3d) and m.weight.dim() == 5]


def apply_conv_plan(module, plan: ConvPlan, *, label: str = "") -> str:
    """Convert `module`'s Conv3d weights to the plan's layout. Returns a one-line record.

    A no-op when the plan does not call for a conversion, so it is safe to call unconditionally.
    """
    if not plan.convert_subgraph:
        return f"{label or 'module'}: no conversion ({plan.backend_name}/{plan.use_layout.value})"
    fmt = plan.use_layout.torch_memory_format()
    convs = convertible_convs(module)
    for m in convs:
        m.to(memory_format=fmt)
    return (f"{label or 'module'}: {len(convs)} Conv3d weights -> {plan.use_layout.value} "
            f"[{plan.tier.name}] via {plan.backend_name}")


def revert_conv_plan(module) -> int:
    """Back to contiguous. Used by A/B probes so an arm cannot leak into the next."""
    convs = convertible_convs(module)
    for m in convs:
        m.to(memory_format=torch.contiguous_format)
    return len(convs)


#: The machine the default timings below were taken on, and the only one they describe. Selecting a
#: convolution backend from these numbers on different silicon is extrapolation, not measurement --
#: cuDNN's kernel coverage for 3D bf16 is exactly what varies between architectures, which is the
#: whole reason this layer exists.
DEFAULTS_MEASURED_ON = (9, 0)      # sm_90, H100 80GB HBM3, torch 2.9 / cuDNN 9.10


#: Cache keyed by (capability, shape) so a load does not re-time what it already knows.
_MEASURED_CACHE: dict = {}


def measure_conv_layouts(*, channels: int = 160, spatial: tuple = (8, 128, 160),
                         iters: int = 20, warmup: int = 5, device=None) -> dict | None:
    """Time the candidate layouts ON THIS DEVICE. Returns what `plan_for_vae` wants, or None.

    THE POINT. `plan_for_vae`'s defaults are timings from one H100, and the backends' own `measure()`
    raises NotImplementedError, so the layout decision was an extrapolation everywhere except the
    machine it was taken on. cuDNN's 3D bf16 kernel coverage is exactly what differs between
    architectures, which is the entire reason this layer exists -- so the honest fix is not to warn
    about the extrapolation, it is to stop extrapolating.

    Conversion is done once outside the timed region because that is how the pass works: weights are
    converted at install, then every forward runs in the chosen layout. Timing a per-call conversion
    would measure a configuration nothing serves.

    Returns None rather than raising when there is no CUDA device, so callers on a laptop fall back to
    the declared defaults and are told they did.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:                                            # noqa: BLE001
        return None

    cap = tuple(getattr(device, "capability", None) or torch.cuda.get_device_capability())
    key = (cap, channels, spatial)
    if key in _MEASURED_CACHE:
        return _MEASURED_CACHE[key]

    import torch
    D, H, W = spatial
    x = torch.randn(1, channels, D, H, W, device="cuda", dtype=torch.bfloat16)
    conv = torch.nn.Conv3d(channels, channels, 3, padding=1, bias=False).cuda().to(torch.bfloat16)

    def time_in(memory_format) -> float:
        c = conv.to(memory_format=memory_format)
        xi = x.to(memory_format=memory_format)
        with torch.no_grad():
            for _ in range(warmup):
                c(xi)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                c(xi)
            e.record()
            torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    out = {("torch_fallback", MemoryLayout.NCDHW): time_in(torch.contiguous_format),
           ("cudnn_conv3d", MemoryLayout.NDHWC): time_in(torch.channels_last_3d)}
    _MEASURED_CACHE[key] = out
    return out


def plan_for_vae(module, *, prefer_bitexact: bool = False, measured: dict | None = None,
                 device=None, measure: bool = False) -> ConvPlan:
    """The plan for a 3D VAE encoder subgraph, asked of the registry rather than asserted.

    `measured` defaults to the encode-scale numbers measured on sm_90, so the caller gets the decision
    that measurement supports; passing fresh numbers is how a different box gets a different answer.

    PROVENANCE IS CHECKED NOW. Nothing ever passed fresh numbers, so every device got the decision an
    H100 supported -- including devices where cuDNN declines a different set of kernels. Pass a probed
    `device` and a mismatch is announced rather than absorbed. It is announced rather than refused on
    purpose: the layout choice may well still be right elsewhere, and silently falling back would trade
    one unverified decision for another. What is not acceptable is not knowing which one you have.
    """
    from instinctflash.backends.conv import REGISTRY, register_declared
    from instinctflash.backends.conv.semantics import ConvSemantics, ConvShape
    register_declared(REGISTRY)
    convs = convertible_convs(module)
    if measured is None and measure:
        measured = measure_conv_layouts(device=device)
    if measured is None:
        measured = {("torch_fallback", MemoryLayout.NCDHW): 175.72,
                    ("cudnn_conv3d", MemoryLayout.NDHWC): 17.00}
        cap = getattr(device, "capability", None)
        if cap is not None and tuple(cap) != DEFAULTS_MEASURED_ON:
            import warnings
            warnings.warn(
                f"conv backend selected from timings measured on sm_"
                f"{DEFAULTS_MEASURED_ON[0]}{DEFAULTS_MEASURED_ON[1]} while running on sm_"
                f"{cap[0]}{cap[1]}. The choice is an extrapolation on this device; measure this "
                f"machine and pass `measured=` to make it a measurement. The accuracy certificate "
                f"behind the NUMERIC tier was also established on sm_"
                f"{DEFAULTS_MEASURED_ON[0]}{DEFAULTS_MEASURED_ON[1]} and does not transfer.",
                stacklevel=2)
    return REGISTRY.select(
        semantics=ConvSemantics.CAUSAL_TIME,
        shape=ConvShape(160, 160, (3, 3, 3), spatial=(8, 128, 160), dtype="bfloat16"),
        have_layout=MemoryLayout.NCDHW, subgraph_size=len(convs),
        prefer_bitexact=prefer_bitexact, measured=measured)


#: The autotune site behind P007. The candidates are the two layouts the conv registry already
#: knows how to select between; what the site adds is measurement ON THIS DEVICE at first load,
#: a persistent per-device cache, an operator override (IFL_AUTOTUNE_VA_CONV_LAYOUT=stock|ndhwc),
#: and a plan line stating what was chosen over what and at which equivalence tier.
CONV_LAYOUT_SITE = register_site(Site(
    name="va_conv_layout",
    candidates=(
        Candidate(
            "stock", Tier.BITEXACT,
            evidence="the incumbent NCDHW/torch dispatch; selecting it changes nothing"),
        Candidate(
            "ndhwc", Tier.NUMERIC,
            evidence=(
                "P007 certificate: paired non-inferiority, margin -0.05 declared before the "
                "run, 555 paired episodes on pinned seeds -- baseline 506/555 = 0.9117, "
                "conv-layout 504/555 = 0.9081, delta -0.0036; exact McNemar two-sided "
                "p = 0.897, one-sided non-inferiority p = 0.00031. NUMERIC because NDHWC "
                "changes the convolution's accumulation order (max|delta| 1.25e-01 on the "
                "encoder output). ESTABLISHED ON sm_90 (H100); the certificate does not "
                "transfer to other silicon, only the latency measurement does"),
            params={"backend": "cudnn_conv3d", "layout": "ndhwc",
                    "certified_on": "sm_90 / 555 paired episodes"}),
    ),
    baseline="stock",
    shape_signature="conv3d 160->160 k3 bf16 1x8x128x160",
))

_LAYOUT_OF = {"stock": torch.contiguous_format, "ndhwc": torch.channels_last_3d}


def conv_layout_bench(*, channels: int = 160, spatial: tuple = (8, 128, 160)):
    """A bench(candidate) -> ms for the conv-layout site: ONE convolution call at encode scale.

    Each arm is built once, OUTSIDE the timed region, in its candidate's memory format -- that is
    how the pass works (weights converted at install, every forward runs in the chosen layout),
    so timing a per-call conversion would measure a configuration nothing serves. The runner
    supplies warmup and the median; this returns one event-timed call per invocation.
    """
    arms: dict = {}

    def bench(cand) -> float:
        fmt = _LAYOUT_OF[cand.name]
        if cand.name not in arms:
            D, H, W = spatial
            x = torch.randn(1, channels, D, H, W, device="cuda",
                            dtype=torch.bfloat16).to(memory_format=fmt)
            conv = torch.nn.Conv3d(channels, channels, 3, padding=1, bias=False
                                   ).cuda().to(torch.bfloat16).to(memory_format=fmt)
            arms[cand.name] = (x, conv)
        x, conv = arms[cand.name]
        with torch.no_grad():
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            conv(x)
            e.record()
            torch.cuda.synchronize()
        return s.elapsed_time(e)

    return bench


def autotune_conv_layout(*, model_id: str = "lingbot-va", prefer_bitexact: bool = False,
                         device=None) -> Decision:
    """Decide the VA conv layout by measurement on THIS device, cached, overridable.

    Replaces the extrapolate-and-warn behaviour for the serving path: `plan_for_vae`'s defaults
    are one H100's timings, and cuDNN's 3D bf16 kernel coverage is exactly what varies between
    architectures. Legality still belongs to the conv registry -- a device where the
    cudnn/NDHWC pair is refused never reaches the bench, it gets the refusal as the reason.

    `prefer_bitexact=True` is the BITEXACT tier ceiling: the NUMERIC candidate is not benched
    and the drop is recorded, exactly the planner's ceiling rule.
    """
    from instinctflash.autotune import _reason  # the one honest way to share the line format

    baseline = CONV_LAYOUT_SITE.candidate("stock")
    if not torch.cuda.is_available():
        return Decision(CONV_LAYOUT_SITE.name, baseline.name, "", 1.0, Tier.BITEXACT,
                        "no-device",
                        reason=_reason(CONV_LAYOUT_SITE, baseline.name, "", 1.0, Tier.BITEXACT,
                                       "no-device", "no CUDA device; the incumbent stands"))

    # Legality first, from the registry that owns it. tier_ceiling here is the LEGALITY ceiling
    # (NUMERIC -- the pair's derived tier); the autotune ceiling below is the CLAIM budget.
    from instinctflash.backends.conv import REGISTRY, register_declared
    from instinctflash.backends.conv.semantics import ConvSemantics, ConvShape
    register_declared(REGISTRY)
    cands = REGISTRY.candidates(
        semantics=ConvSemantics.CAUSAL_TIME,
        shape=ConvShape(160, 160, (3, 3, 3), spatial=(8, 128, 160), dtype="bfloat16"),
        have_layout=MemoryLayout.NCDHW, tier_ceiling=Tier.NUMERIC, subgraph_size=62)
    ndhwc = next((c for c in cands
                  if c.backend_name == "cudnn_conv3d" and c.use_layout is MemoryLayout.NDHWC),
                 None)
    if ndhwc is None or not ndhwc.legal:
        why = ndhwc.verdict.reason if ndhwc is not None else "no cudnn_conv3d/NDHWC pair registered"
        return Decision(CONV_LAYOUT_SITE.name, baseline.name, "", 1.0, Tier.BITEXACT, "ceiling",
                        reason=_reason(CONV_LAYOUT_SITE, baseline.name, "", 1.0, Tier.BITEXACT,
                                       "ceiling", f"cudnn/NDHWC refused by the conv registry: "
                                                  f"{why}"))

    from instinctflash.passes.contract import DeviceProfile
    if device is None:
        device = DeviceProfile.probe()
    return autotune(
        CONV_LAYOUT_SITE, conv_layout_bench(), model_id=model_id, device=device,
        tier_ceiling=Tier.BITEXACT if prefer_bitexact else Tier.NUMERIC,
        n=7, warmup=3)


def conv_plan_from_decision(decision: Decision) -> ConvPlan:
    """The ConvPlan a Decision denotes. Applying it stays `apply_conv_plan`'s job."""
    if decision.chosen == "ndhwc":
        return ConvPlan("cudnn_conv3d", MemoryLayout.NDHWC, True, Tier.NUMERIC, decision.reason)
    return ConvPlan("torch_fallback", MemoryLayout.NCDHW, False, Tier.BITEXACT, decision.reason)


def install_conv_layout(server, *, prefer_bitexact: bool = False, model_id: str = "lingbot-va",
                        plan=None) -> list[str]:
    """Apply the conv plan to EVERY VAE subgraph a LingBot-VA server owns.

    Enumerated by attribute name rather than by walking the server, because a VAE that is missed is
    silent: the run simply comes out slower than it should and nothing reports why.

    THE DECISION IS AUTOTUNED: measured on this device at first load, cached at
    ~/.cache/instinctflash/autotune.json, forceable with IFL_AUTOTUNE_VA_CONV_LAYOUT and
    disabled entirely (baseline layout) with IFL_AUTOTUNE=0. One decision serves both VAEs --
    they share the conv signature, and converting only one is the half-applied state the
    docstring above exists to prevent. Pass `plan=` (a planners.Plan) and the decision is
    recorded there, so explain() shows the swap and Plan.tier() prices it.
    """
    decision = autotune_conv_layout(model_id=model_id, prefer_bitexact=prefer_bitexact)
    conv_plan = conv_plan_from_decision(decision)
    out = [decision.reason]
    if plan is not None:
        from instinctflash.autotune import record_decision
        record_decision(plan, decision)
    applied_any = False
    for attr in ("streaming_vae", "streaming_vae_half"):
        sv = getattr(server, attr, None)
        if sv is None:
            continue
        vae = getattr(sv, "vae", sv)
        out.append(apply_conv_plan(vae, conv_plan, label=attr))
        applied_any = True
    if not applied_any:
        out.append("no VAE subgraph found; conv layout not applied")
    return out
