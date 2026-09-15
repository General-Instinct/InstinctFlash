"""FP8 W+A wiring for pi05, opt-in via IFL_PI05_FP8=1.

Order is load-bearing: fold norms (fp32) -> swap linears to calibration-mode FP8 -> run the
caller's calibration passes (eager) -> freeze static scales -> only THEN install the static
capture, so the captured graph bakes the fp8 hot path and never sees the calibration branch.

Scope: the large GEMMs of vision tower, language model and action expert. Norm folding applies
only to the language model's static GemmaRMSNorms; the expert's norms are time-conditioned
(AdaRMS-style) and the vision tower uses LayerNorm — neither folds into a constant gain.
"""

from __future__ import annotations

from instinctflash.passes.fp8_linear import apply_fp8, fold_gemma_norms, freeze_fp8

PREFIX = "paligemma_with_expert"

INCLUDE = [
    f"{PREFIX}.paligemma.model.vision_tower.*self_attn.*_proj",
    f"{PREFIX}.paligemma.model.vision_tower.*mlp.fc*",
    f"{PREFIX}.paligemma.model.multi_modal_projector.linear",
    f"{PREFIX}.paligemma.model.language_model.layers.*self_attn.*_proj",
    f"{PREFIX}.paligemma.model.language_model.layers.*mlp.*_proj",
    f"{PREFIX}.gemma_expert.model.layers.*self_attn.*_proj",
    f"{PREFIX}.gemma_expert.model.layers.*mlp.*_proj",
]

LM_LAYER_PATTERN = f"{PREFIX}.paligemma.model.language_model.layers.*"


def install_fp8(flow_model, calib_fn, *, fold_norms: bool = True) -> dict:
    """Quantize, calibrate with `calib_fn(model)` (run >=8 varied full passes), freeze."""
    folded = fold_gemma_norms(flow_model, LM_LAYER_PATTERN) if fold_norms else 0
    wrapped = apply_fp8(flow_model, INCLUDE, merge_gate_up=True)
    calib_fn(flow_model)
    frozen = freeze_fp8(flow_model)
    return {"folded_layers": folded, "wrapped": len(wrapped), "frozen": frozen}
