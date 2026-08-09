---
license: apache-2.0
library_name: instinctwm
tags:
  - world-action-model
  - robotics
---

# wm-blockheads-2v4a — an example InstinctWM checkpoint

This is a **layout example**, not real weights (`model.safetensors` is a single null byte). It exists
so a checkpoint author can see exactly what a publishable package looks like, and so
`tests/test_example_checkpoint.py` can validate a real artifact rather than a fixture invented inside
the test.

## Validate it

```bash
python -m instinctwm.descriptors.package examples/checkpoint/wm-blockheads-2v4a
```

```
examples/checkpoint/wm-blockheads-2v4a
  servable package: YES
  model_id='example-org/wm-blockheads-2v4a' backbone='wan_va' servable=True
  output_projection: per_interval_velocity_heads n_intervals=8 block=4 convention=sigma_descending
```

## What the runtime reads

Only the `execution` block of `instinctwm.json`. That is the entire contract:

| field | why the runtime needs it |
|:--|:--|
| `model_id` | identity, and the key the plan is reported against |
| `backbone` | which adapter can drive these weights |
| `servable` | **the one gate.** Fit to serve, or not. The *reason* is a training fact and lives in provenance |
| `guidance` | per stream: `cfg`, `positive_only`, `none` — decides whether a batch is duplicated |
| `nfe` | forwards per stream at the intended operating point |
| `output_projection` | the capability that replaces "this is a distilled checkpoint" |

## `output_projection` is the interesting one

```json
"output_projection": {
  "kind": "per_interval_velocity_heads",
  "n_intervals": 8, "block": 4,
  "velocity_convention": "sigma_descending",
  "foldable": true
}
```

This says: *the final projection provides L linear heads per block over an N-interval grid, the heads
emit velocity in the σ-descending convention, and they are foldable into a single affine map at load
time.* Those are facts about **what the weights are**, checkable by looking at them.

They are not facts about how the weights were produced. A checkpoint distilled by DMD2, by LCM, by
consistency training, or by a method that does not exist yet declares the *same three numbers* and is
served by the *same code path*. That is the whole design.

`velocity_convention` is a field rather than a comment for a specific reason: a double sign flip there
once produced 0/100 on RoboTwin against a 92/100 control, and it was diagnosed only by reading the
training loss's sign convention. A comment cannot be checked; a field can.

## What the runtime never reads

The `provenance` block. `load_declaration()` parses it only to drop it, and
`tests/test_checkpoint_platform.py` asserts that two checkpoints with identical `execution` and
opposite `provenance` produce a **byte-identical plan**.

**You can delete `provenance` entirely and this checkpoint still serves.** Verify before publishing:

```python
from instinctwm.descriptors.package import publishability
ok, findings = publishability("examples/checkpoint/wm-blockheads-2v4a")
```

If `ok` is false, something the runtime needs is in the wrong namespace — which is exactly the mistake
the two-namespace split exists to catch.

## Serving it

```python
from instinctwm.descriptors.package import from_pretrained

ckpt = from_pretrained("example-org/wm-blockheads-2v4a")   # or a local path
print(ckpt.capabilities())
# frozenset({'servable', 'backbone:wan_va',
#            'output_projection:per_interval_velocity_heads',
#            'output_projection:foldable',
#            'guidance:video=cfg', 'guidance:action=positive_only'})
```

Those tokens are the *only* thing the planner is given about your checkpoint. A pass that declares
`requires_capabilities` is admitted when its tokens are present and skipped when they are not — and a
pass that declares none composes with every checkpoint, which is the default.
