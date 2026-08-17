# A VLA family, to test whether the abstraction is real

`lerobot/pi05_base` is a vision-language-action policy. It is here because it is structurally unlike
LingBot-VA in almost every way that the runtime cares about, which makes it a test of whether
InstinctWM's declarations describe *execution* or merely describe one world model.

| | LingBot-VA (world-action) | pi05 (VLA) |
|:--|:--|:--|
| streams | two coupled: video + action | one: action |
| observation history | growing ring, 72-frame window | `n_obs_steps=1`, a single observation |
| K/V across control steps | carried and **grown** | prefix, **recomputed every step** |
| K/V lifetime | `EPISODE` | `CHUNK` |
| guidance | CFG at 5.0 on video | none — flow matching |
| forwards per control step | 10 at the shipped schedule | 11 (1 prefix + 10 flow steps) |
| action chunk | 32 | 50 |
| commit phase | yes, a deferred ring advance | none |

Every fact in the right-hand column is read from the checkpoint's own `config.json`, not guessed.

## What it showed

**The generic pass fired, on its own declared merits.** `conditioning_prefill` reports
`['prefix_kv'] declared pure in ['chunk'] scope, but recomputed on all 11 forwards per control step`.
Nothing about that pass knows what a VLA is; it read `PurityKey(scope=CHUNK)` and the phase list. That
is the abstraction doing its job across families.

**`cfg_branch_elision` and `obs_decode_elision` declined for the right reasons** — no stream requests
CFG, and a VLA predicts no pixels.

**Two defects surfaced, both now fixed.** Planning this model reported `APPLY` on three passes that
rewrite the LingBot-VA *server object*, with no indication their applicability had not been checked;
`capabilities=None` is inspection mode and deliberately does not filter, so the planner now annotates
those results `APPLICABILITY UNCHECKED` instead of silently endorsing them. And
`conditioning_prefill` quoted LingBot-VA's "89 of 226 TFLOP … 360 MiB resident" as this model's
expectation; a cross-family expectation now names the model it was measured on.

**It also explains, from the model side, why static graph capture pays for VLAs and not for us.**
`n_obs_steps=1` means nothing accumulates between control steps, so every tensor shape is fixed and a
captured graph stays valid. LingBot-VA accumulates 152 slots per cycle. Hand-tuned engines that
capture a whole Pi0-class forward are exploiting that property of the model, not out-optimising the
runtime.

## Run it

```bash
pip install ./examples/pi05_vla
```

```python
from instinctwm import Optimizer, load
print(Optimizer().compile(load("pi05").spec()).explain())
```

Inference needs the 14.5 GB checkpoint and a GPU, and is not part of this example: what is being
tested here is whether the declaration and the planner survive a second model family, which needs
neither.

## Attribution

`lerobot/pi05_base` and LeRobot are Apache-2.0, © The HuggingFace Inc. team. Nothing is copied here —
this adapter reads that checkpoint's published configuration and declares it in InstinctWM's terms.
