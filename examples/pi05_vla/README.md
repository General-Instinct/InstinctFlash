# pi05, a VLA in InstinctFlash

`lerobot/pi05_base` is a vision-language-action policy, registered from outside the core through the
`instinctflash.adapters` entry point. It is here because it is structurally unlike LingBot-VA in almost
every way the runtime cares about, which makes it a test of whether InstinctFlash's declarations describe
*execution* or merely describe one world model.

| | LingBot-VA (world-action) | pi05 (VLA) |
|:--|:--|:--|
| streams | two coupled: video + action | one: a prefix |
| observation history | growing ring, 72-frame window | `n_obs_steps=1`, a single observation |
| K/V across control steps | carried and **grown** | prefix, **recomputed every step** |
| K/V lifetime | `EPISODE` | `CHUNK` |
| guidance | CFG at 5.0 on video | none — flow matching |
| forwards per control step | 10 at the shipped schedule | 11 (1 prefix + 10 flow steps) |
| action chunk | 32 | 50 |
| commit phase | yes, a deferred ring advance | none |
| language | a prompt, encoded once per episode | a prompt, **tokenized by a processor pipeline** |

Every fact in the right-hand column is read from that checkpoint's own `config.json`, not guessed:
three cameras at `(3,224,224)`, a 32-dimensional state, `num_inference_steps: 10`, `chunk_size: 50`.

## What pi05 needs that a world model does not

**A processor pipeline, and it is not optional.** `predict_action_chunk` reads
`batch[OBS_LANGUAGE_TOKENS]` and `batch[OBS_LANGUAGE_ATTENTION_MASK]` — already tokenized. Text never
reaches the policy. The tokenizer, the input normalisation and the action un-normalisation all live in
a `PolicyProcessorPipeline` published beside the weights as `policy_preprocessor.json`. A VLA served
without it is not slow, it is **wrong**: fed unnormalised pixels, returning actions in a normalised
space nobody can execute. `build_in_process` therefore loads the policy *and* its pipeline, and maps
the declaration's `prompt` onto LeRobot's `task` key — model semantics, so it stays in the adapter.

**A patched `transformers`.** pi05 asserts
`transformers.models.siglip.check.check_whether_transformers_replace_is_installed_correctly()`, which
standard transformers does not provide. Upstream ships the replacement files in openpi's
`transformers_replace/`, and newer LeRobot exposes them as `pip install "lerobot[pi]"` — an extra that
`lerobot 0.4.4` does not have. Until that environment exists, `build_in_process` raises with the real
reason instead of a `ValueError` about a version.

## The concept this comparison contributed

Three families made one property visible that none of them declares directly: whether tensor shapes
repeat from one control cycle to the next. A stream that outlives a cycle accumulates, so the extent
read on cycle N differs from cycle N-1 and a captured graph is invalid. That is derivable from the
declared stream lifetimes, so `AdapterSpec.shapes_static_across_cycles()` derives it:

    LingBot-VA   GROWS    streams ['action', 'video'] outlive a control cycle
    pi05         STATIC   all streams (prefix) are rebuilt within a control cycle

Whole-cycle graph capture measured **1.43x slower** on LingBot-VA and is the headline optimization of
hand-tuned VLA engines. Both are consequences of that one line.

## Graph capture is the default, and the self-check is the reason it can be

On capture-capable devices (CUDA with graph support; the measured bandwidth-bound-edge class
declines at plan time with its 1.04x reason), every pi05-class checkpoint — `pi05_base`, published
fine-tunes, and a directory fresh out of `lerobot-train` — serves on the replay-safe **static-KV
CUDA-graph capture** by default (`pi05_iwm/static_capture.py`; measured 206.7 → 72.8 ms/chunk on
H100/v044). No flag, no `compile_model` trigger. It used to be opt-in because capture was gated on
evidence measured on *other* checkpoints; what makes the default safe is that the evidence is now
re-earned **per process**:

- **The bit-exact self-check.** Immediately after the first capture, replay is compared against
  upstream eager `denoise_step` on staged inputs the capture never saw — fresh `x_t` draws from a
  dedicated generator, every warmed schedule timestep, and a synthetically *refilled* prefix so a
  graph that baked K/V values instead of reading the live buffers cannot pass. Exact equality
  (`atol=0`). PASS → replay serves, and the plan's `graph_capture` entry gains the line
  `self-check bit-exact on N inputs`. FAIL → the graphs are released, `denoise_step` is rebound to
  upstream, the observed delta is printed and recorded on the plan, and serving continues on eager
  arithmetic. The check costs seconds, once per process, at first capture.
- **Kill-switch:** `IFL_PI05_NO_CAPTURE=1` serves eager (recorded on the plan, printed). It is
  refused on checkpoints that *declare* the TF32 static-KV operating point, because there eager
  would be a different execution semantics than the declared one.
- **Retired opt-ins:** `IFL_PI05_STATIC_CAPTURE=1` (now the default) and `IFL_PI05_CAPTURE=1`
  (the DynamicCache experiment, measured replay-unsafe) are no-ops with a notice.

## Run it

```bash
pip install ./examples/pi05_vla
instinctflash plan  <a-checkpoint-declaring-backbone-pi05>
instinctflash run   <a-checkpoint-declaring-backbone-pi05>
```

`plan` needs no weights and no GPU. `run` needs the patched transformers described above, plus a GPU
with room for 14.5 GB of weights.

## Attribution

`lerobot/pi05_base` and LeRobot are Apache-2.0, © The HuggingFace Inc. team. Nothing is copied here —
this adapter reads that checkpoint's published configuration and declares it in InstinctFlash's terms.
