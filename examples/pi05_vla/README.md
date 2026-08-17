# Two VLA families, to test whether the abstraction is real

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

A third column, ACT (`lerobot/act_aloha_sim_transfer_cube_human`), is the degenerate case: an
encoder-decoder transformer that emits a 100-action chunk in **one forward**, with no refinement loop,
no guidance and no persistent stream at all. If `nfe` were secretly a diffusion concept rather than an
execution one, ACT is where that would show.

Every fact in these columns is read from the checkpoints' own `config.json`, not guessed.

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

## The concept the comparison was missing

Three families made one property visible that none of them declares directly: whether tensor shapes
repeat from one control cycle to the next. A stream that outlives a cycle accumulates, so the extent
read on cycle N differs from cycle N-1 and a captured graph is invalid. That is derivable from the
declared stream lifetimes, so `AdapterSpec.shapes_static_across_cycles()` now derives it:

    LingBot-VA   GROWS    streams ['action', 'video'] outlive a control cycle
    pi05         STATIC   all streams (prefix) are rebuilt within a control cycle
    ACT          STATIC   no stream persists

Whole-cycle graph capture measured **1.43x slower** on LingBot-VA and is the headline optimization of
hand-tuned VLA engines. Both are consequences of that one line. Before it was derivable, the only way
to find out was to build the pass and measure the regression.

## Run it

```bash
pip install ./examples/pi05_vla
```

ACT runs end to end, from a declaration alone:

```bash
python examples/pi05_vla/run_act_end_to_end.py
```

    describe()            backbone act, servable, nfe {'action': 1}
    from_pretrained()     plan compiled, shapes static across cycles
    episode.predict() x5  action (14,) float32, finite, five consecutive cycles

The package it builds carries **no weights at all** — `execution.base_weights` names the upstream
LeRobot repo and the adapter resolves it at load. That is the shape a third party adopting someone
else's checkpoint actually needs, and it did not work before this example existed: `validate_package`
demanded local weight files even when a pointer was declared, so describing somebody else's
checkpoint required copying their gigabytes first.

LeRobot's `ACTPolicy.select_action` hides an internal action-chunk queue and returns one action per
call; `reset` clears it. Our `predict`/`reset` map onto that with nothing model-specific reaching the
runtime — two systems independently deciding that action-chunk buffering belongs behind one verb.

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
