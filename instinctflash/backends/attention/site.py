"""How an adapter publishes an attention site, and how a backend reads one.

THE SEAM, STATED ONCE MORE

    the adapter says   "layer 7 self-attention computes SOFTMAX_FULL over BSHD, ring-addressed KV,
                        40 heads of dim 128, seq_q 1 and seq_kv growing to 9792, bf16, in the
                        video phase which runs 25 forwards per cycle"

    the backend says   "I implement SOFTMAX_FULL over BSHD with causal or windowed flags, head_dim
                        multiple of 8 up to 256, and I cannot take a data-dependent dense mask"

    the planner        intersects the two, and neither side has learned anything about the other

`attention_site()` builds the `Site` so that the required attribute names exist in one place rather
than being spelled slightly differently by each adapter. `read_site()` is its inverse, so a backend
never indexes `attrs` by hand. A pass that needs a property not in this vocabulary should ADD it here
-- the alternative, reaching into the model, is the thing the seam exists to prevent.

WHY `forwards_per_cycle` IS ON THE SITE

Because profitability is meaningless without it. Attention is a PER_STEP cost, so what a backend can
win scales with how many times the site is entered per control cycle -- 75 at Quality, 6 at Fast. That
number comes from the checkpoint's declared `phases`, which is exactly why an operating point is a
descriptor delta and not a runtime mode. A site that omitted it would force the planner to guess the
denominator, and guessing it is how graph capture came to be enabled at an operating point where it is
a 2x regression.
"""

from __future__ import annotations

from dataclasses import dataclass

from instinctflash.backends.attention.semantics import (
    AttentionSemantics,
    AttentionShape,
    MaskKind,
    MaskSpec,
    QKVLayout,
)
from instinctflash.passes.interface import Site, SiteKind
from instinctflash.runtime.state.types import Addressing


@dataclass(frozen=True)
class AttentionSiteFacts:
    """The typed view of an ATTENTION site's `attrs`. What every backend is allowed to read."""

    semantics: AttentionSemantics
    mask: MaskSpec
    layout: QKVLayout
    addressing: Addressing
    shape: AttentionShape
    #: which declared phase this site runs in, and how many forwards that phase costs per cycle.
    #: Both come from `AdapterSpec.phases`; neither is a runtime flag.
    phase: str = ""
    forwards_per_cycle: int = 1
    #: self- vs cross-attention. Cross-attention over episode-constant conditioning is the site
    #: `conditioning_prefill` (P002) already hoists, so a backend competing there is competing with
    #: a pass that removed the work entirely -- worth knowing before measuring.
    is_cross_attention: bool = False
    stream: str = ""


def attention_site(
    site_id: str,
    *,
    semantics: AttentionSemantics,
    mask: MaskSpec,
    layout: QKVLayout,
    addressing: Addressing,
    shape: AttentionShape,
    phase: str = "",
    forwards_per_cycle: int = 1,
    is_cross_attention: bool = False,
    stream: str = "",
    handle=None,
) -> Site:
    """Build an ATTENTION `Site`. Called by adapters; `handle` is the adapter's own rewrite handle.

    The handle is opaque to every backend and to the planner. Only the executor passes it back to the
    adapter, which is what keeps a backend free of model symbols.
    """
    return Site(
        kind=SiteKind.ATTENTION,
        id=site_id,
        attrs={
            "semantics": semantics,
            "mask": mask,
            "layout": layout,
            "addressing": addressing,
            "shape": shape,
            "phase": phase,
            "forwards_per_cycle": forwards_per_cycle,
            "is_cross_attention": is_cross_attention,
            "stream": stream,
            "handle": handle,
        },
    )


def read_site(site: Site) -> AttentionSiteFacts:
    """Typed read of an ATTENTION site. Raises on a malformed one rather than defaulting.

    Defaulting a missing `semantics` would be the worst possible failure mode in this layer: it would
    let a site be served by a backend computing a different function, which is precisely what
    `semantics.py` exists to make impossible.
    """
    if site.kind is not SiteKind.ATTENTION:
        raise ValueError(f"{site.id}: not an ATTENTION site (kind={site.kind.value})")
    a = site.attrs
    missing = [k for k in ("semantics", "mask", "layout", "addressing", "shape") if k not in a]
    if missing:
        raise ValueError(
            f"{site.id}: ATTENTION site is missing {missing}. An adapter must declare what function "
            f"the site computes; there is no safe default.")
    if not isinstance(a["semantics"], AttentionSemantics):
        raise TypeError(f"{site.id}: semantics must be an AttentionSemantics, got "
                        f"{type(a['semantics']).__name__}")
    return AttentionSiteFacts(
        semantics=a["semantics"], mask=a["mask"], layout=a["layout"],
        addressing=a["addressing"], shape=a["shape"],
        phase=a.get("phase", ""), forwards_per_cycle=int(a.get("forwards_per_cycle", 1)),
        is_cross_attention=bool(a.get("is_cross_attention", False)),
        stream=a.get("stream", ""),
    )


#: The site LingBot-VA's video trunk would publish, as a worked example. NOT wired into the adapter:
#: this is the architecture step, and `lingbot.sites()` is unchanged. It is here so the vocabulary can
#: be checked against a real model's real numbers rather than an invented one.
#:
#: THE MASK FIELD IS THE WHOLE POINT, and writing it down corrected a misconception. Under P003 the
#: live KV set is the contiguous interval `[start, start+count)`, so the site attends over a SLICE and
#: needs NO MASK -- `MaskKind.NONE`. The mask only ever existed to select the live set out of a padded
#: buffer, and the stock path built it with `mask.nonzero()` per layer per forward:
#: DENSE_DATA_DEPENDENT, which rules out every flash-family backend and is also what raised
#: `cudaErrorStreamCaptureInvalidated` under graph capture.
#:
#: So one addressing change moved this site from "no fast attention backend is legal" to "almost all
#: are". That is worth stating precisely because it inverts the intuitive order of work: the
#: prerequisite for Layer 4 was a Layer 3 change that has already shipped.
def lingbot_video_self_attention_example(forwards_per_cycle: int = 25) -> Site:
    return attention_site(
        "example.lingbot.video.self_attn",
        semantics=AttentionSemantics.SOFTMAX_FULL,
        mask=MaskSpec(kind=MaskKind.NONE, materialised=False),
        layout=QKVLayout.BSHD,
        addressing=Addressing.RING_INTERVAL,
        shape=AttentionShape(
            n_heads=40, head_dim=128, seq_q_min=1, seq_q_max=272,
            seq_kv_min=272, seq_kv_max=9792, dtype="bfloat16"),
        phase="video",
        forwards_per_cycle=forwards_per_cycle,
        stream="video",
    )
