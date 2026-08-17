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

from instinctwm.backends.conv.registry import ConvPlan
from instinctwm.backends.conv.semantics import MemoryLayout


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


def plan_for_vae(module, *, prefer_bitexact: bool = False, measured: dict | None = None,
                 device=None) -> ConvPlan:
    """The plan for a 3D VAE encoder subgraph, asked of the registry rather than asserted.

    `measured` defaults to the encode-scale numbers measured on sm_90, so the caller gets the decision
    that measurement supports; passing fresh numbers is how a different box gets a different answer.

    PROVENANCE IS CHECKED NOW. Nothing ever passed fresh numbers, so every device got the decision an
    H100 supported -- including devices where cuDNN declines a different set of kernels. Pass a probed
    `device` and a mismatch is announced rather than absorbed. It is announced rather than refused on
    purpose: the layout choice may well still be right elsewhere, and silently falling back would trade
    one unverified decision for another. What is not acceptable is not knowing which one you have.
    """
    from instinctwm.backends.conv import REGISTRY, register_declared
    from instinctwm.backends.conv.semantics import ConvSemantics, ConvShape
    register_declared(REGISTRY)
    convs = convertible_convs(module)
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


def install_conv_layout(server, *, prefer_bitexact: bool = False) -> list[str]:
    """Apply the conv plan to EVERY VAE subgraph a LingBot-VA server owns.

    Enumerated by attribute name rather than by walking the server, because a VAE that is missed is
    silent: the run simply comes out slower than it should and nothing reports why.
    """
    out = []
    for attr in ("streaming_vae", "streaming_vae_half"):
        sv = getattr(server, attr, None)
        if sv is None:
            continue
        vae = getattr(sv, "vae", sv)
        plan = plan_for_vae(vae, prefer_bitexact=prefer_bitexact)
        out.append(apply_conv_plan(vae, plan, label=attr))
    if not out:
        out.append("no VAE subgraph found; conv layout not applied")
    return out
