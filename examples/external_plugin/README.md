# A new model family, with no changes to InstinctWM

`gridworld-ar` is a toy world-action model chosen to be structurally *unlike* LingBot-VA, so that
integrating it exercises the parts of the runtime a second model family will actually hit:

| | LingBot-VA | gridworld-ar |
|:--|:--|:--|
| streams | two (video, action) | none declared |
| control cycle | diffusion, 79 forwards | autoregressive, 1 forward |
| guidance | classifier-free | none |
| episode state | ring KV with a commit phase | a token history |
| frozen components | VAE + T5 by Hub pointer | none |

## The whole integration

```bash
pip install ./examples/external_plugin      # or your own package, from anywhere
```

```python
from instinctwm import Runtime               # note: no import of gridworld_wm
runtime = Runtime.from_pretrained("path/to/my-world-model")
with runtime.episode() as ep:
    action = ep.predict({"obs": [0.1, 0.2, 0.3]})
```

The binding is one entry point in the author's own `pyproject.toml`:

```toml
[project.entry-points."instinctwm.adapters"]
gridworld_ar = "gridworld_wm.adapter:GridworldAdapter"
```

InstinctWM discovers it from installed metadata, so a checkpoint that declares
`"backbone": "gridworld_ar"` resolves with no import, no import-order rule, and no PR.

## What the author has to write

Two methods, 80 lines of adapter, 45 of model:

- `spec()` → an `AdapterSpec` of facts. Empty `streams`, `guidance` and `purity` are fine; declaring
  nothing means the optimizer offers nothing, which is the correct default for a new family.
- `build_in_process(checkpoint, plan, *, device, nfe)` → any object with `predict(observation)`.
  Optional on that object: `reset(**conditioning)`, `commit(observation, action)`, `close()`.

`commit` is how a model with per-cycle state updates stays loopable without the *user* learning that
phases exist. `gridworld-ar` does not need one.

## Run it

```bash
cd examples/external_plugin
PYTHONPATH=. python build_checkpoint.py     # writes ./my-world-model (35 KB)
python run_as_user.py                       # describe -> from_pretrained -> closed loop
```

## Deliberately not required

No InstinctWM source change, no pass, no planner knowledge, no entry in any InstinctWM registry
file, and nothing model-specific in `instinctwm.json` — `vocab`, `dim` and `history` live in the
author's own `config.json`, because they are model knowledge and the declaration is for execution
facts.
