---
license: apache-2.0
library_name: instinctwm
tags:
  - world-action-model
  - robotics
---

# tiny-wam-2v2a — a real, runnable InstinctWM checkpoint

**327 KB of real weights.** Not a fixture: `model.safetensors` loads, the module runs a forward pass,
and it produces an action chunk. It is published so the whole user-facing workflow can be executed
end to end in a few seconds on a laptop.

**No accuracy claim.** The weights are seeded random and the model is not trained. It predicts
nothing useful. What is demonstrated is the *workflow*: declare → publish → resolve → plan → run.

## Run it

```bash
python examples/tiny_wam/build_checkpoint.py       # regenerate (deterministic, seed 0)
PYTHONPATH=. python examples/tiny_wam/run_end_to_end.py
```

Nine steps, each checked:

| step | |
|--:|:--|
| 1 | the package validates against the published layout |
| 2 | `publishability()` — servable with `provenance` stripped |
| 3 | `from_pretrained()` loads it |
| 4 | its `backbone` resolves to a **third-party** adapter, registered at call time |
| 5 | `capabilities()` derives tokens from the execution block alone |
| 6 | the adapter describes the control step |
| 7 | `Optimizer.compile()` produces a Plan **from those capabilities** |
| 8 | every applied pass is installed or shown vacuous |
| 9 | one real inference, and `output_projection.foldable` is *verified* against the weights |

## What makes this a real test of the platform

**The adapter is not built in.** `examples/tiny_wam/adapter.py` lives outside `instinctwm/`, nothing
in the runtime imports it, and step 4 asserts `tiny-wam` is absent from `available_models()` *before*
registering it. An author with their own backbone does exactly this.

**The declared capabilities are checkable against the weights.** `output_projection.foldable: true`
claims the L linear heads fold into one affine map. Step 9 runs the model both ways and compares:

```
output_projection.foldable is TRUE of these weights: folded == unfolded   max|delta| = 5.960e-08
```

That is the difference between a declaration and a promise.

**Substrate passes apply to every checkpoint.** `fsdp_elision`, `allocator_churn_elision` and
`debug_dump_elision` describe the serving *environment*, not the model, so they appear in the plan for
any checkpoint. The adapter must account for each — install it, or show it is vacuous here — and
refuses anything it can do neither with. This example runs in-process with no FSDP wrapper, no
per-step `empty_cache()` and no debug-dump hook, so all three are vacuous.

## The declaration

```jsonc
"execution": {
  "model_id": "example-org/tiny-wam-2v2a",
  "backbone": "tiny-wam",          // must name a REGISTERED adapter
  "servable": true,
  "guidance": {"video": "cfg", "action": "positive_only"},
  "nfe": {"video": 2, "action": 2},
  "output_projection": {
    "kind": "per_interval_velocity_heads",
    "n_intervals": 8, "block": 4,
    "velocity_convention": "sigma_descending",
    "foldable": true
  },
  "param_bytes": 323008
}
```

Every field is true of the module in `examples/tiny_wam/model.py`, and the ones that can be checked
against the weights are checked in step 9.

## Scope

`backbone: "tiny-wam"` must resolve to a registered adapter or the checkpoint is not servable, however
well it declares itself. **Many checkpoints per backbone is supported today; arbitrary new backbones
without an adapter is a separate future extension.** See the "Scope" section of the top-level README.
