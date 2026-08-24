"""Whether whole-forward graph capture is legal for a model, decided from its declaration.

TWO PASSES SHARE THE NAME `graph_capture`, AND THAT IS DELIBERATE.

  * this one, a PLANNER pass: decides IF capture can pay, from `AdapterSpec` alone -- no torch, no
    weights, no GPU, so `instinctflash plan <model-id>` answers it before anything is downloaded.
  * `instinctflash/passes/graph_capture.py`, an ENGINE pass: does the capturing, against the sites an
    adapter publishes.

They were split by necessity, not taste. The engine pass has existed and been generic for a while, and
the planner could not plan it -- so a model's plan never mentioned capture, `install()` had nothing to
act on, and the only way to find out whether capture was possible was to build it and measure. pi05 was
capturable the whole time and no plan said so. One name, two roles: the planner decides, the engine
acts, and a reader comparing `instinctflash plan` against what the runtime prints sees the same word.

WHY A DECLARATION IS ENOUGH TO DECIDE THIS

A captured graph is valid only while shapes repeat, and whether they repeat follows from stream
lifetimes, which the adapter declares. `AdapterSpec.shapes_static_across_cycles()` derives it. The
two families measured here land on opposite sides, and neither answer was chosen:

    LingBot-VA   EPISODE-lifetime streams; ring KV grows 152 slots/cycle, saturates near cycle 64,
                 real episodes are a median of 15.6 cycles. Capture measured 1.43x SLOWER.
    pi05         one CHUNK-lifetime prefix rebuilt per chunk, so shapes repeat and this pass APPLIES.

WHAT APPLYING DOES NOT MEAN. On pi05 the declaration is right and capture still does not ship: the
region replays 1.55x faster and computes the WRONG answer for any input it was not captured from, by
up to 48% of the signal, because `denoise_step` appends 50 entries to a `DynamicCache` it creates
inside the region. The engine pass now detects that by comparing replay against eager on a second
input and discards the graph. So this pass says "shapes repeat, capture is worth attempting", which is
all a declaration can support -- whether a specific region is replay-SAFE is a measurement, and it is
taken where the capture happens.
"""

from __future__ import annotations

from instinctflash.adapters.base import AdapterSpec
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import HardwareReq
from instinctflash.planners.planner import PassResult, Tier


class GraphCaptureApplicable:
    """Fires when the declaration says shapes repeat across control cycles."""

    name = "graph_capture"

    #: Capture is a CUDA facility. Stated as a requirement so the planner enforces it rather than
    #: the pass discovering it at build time on a CPU-only machine.
    hardware = HardwareReq(requires=("cuda", "cuda_graphs"))

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        static, why = spec.shapes_static_across_cycles()
        if not static:
            return PassResult(self.name, False, Tier.BITEXACT,
                              f"{why}, so a captured graph is invalidated every cycle and recapture "
                              f"costs more than replay saves (measured 1.43x SLOWER on LingBot-VA)")

        n_fwd = spec.total_forwards()
        return PassResult(
            name=self.name,
            applies=True,
            tier=Tier.BITEXACT,
            reason=(f"{why}, so one capture serves every cycle; {n_fwd} forward(s) per control step "
                    f"({spec.forwards_breakdown()}) each pay full dispatch cost today"),
            params={"capture_unit_from": "adapter sites of kind CAPTURE_UNIT",
                    "verify": "host-effect gate refuses a region that mutates host state"},
            # ATTRIBUTED. This pass is generic and will fire on models these numbers were never taken
            # on, so the number names the model it came from. Presenting pi05's 3.16x as a forecast
            # for an unseen backbone is the failure mode conditioning_prefill already had once.
            expected_win=(
                "removes per-forward dispatch cost where the region is replay-safe. Measured ON pi05 "
                "(H100, pi05_base, 50-step chunk): a denoise step is 2187 kernel launches, 17.25 ms "
                "of CPU submit against 9.48 ms of GPU work -- 99.9% submit-bound, the GPU idle ~45% "
                "of the step. The naive region (per-step DynamicCache clone+append) replays fast and "
                "WRONG and stays discarded; the static-KV region (pi05_iwm/static_capture.py) is "
                "replay-safe by construction and collected the prize: denoise step 16.25 -> 4.57 ms "
                "(3.55x), chunk 298.7 -> 181.3 ms (1.65x), bitexact on unseen inputs and prompts. "
                "Both halves are this model's -- a backbone whose forwards are already compute-bound "
                "has nothing here to win either way, so measure the submit-vs-busy split, then "
                "measure replay against eager on a SECOND input"),
        )
