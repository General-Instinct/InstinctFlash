# Cosmos3 action policies (Edge / Nano), in InstinctFlash

`nvidia/Cosmos3-Edge-Policy-DROID` (3.86B) and `nvidia/Cosmos3-Nano-Policy-DROID` (15.75B) are
two-tower MoT action policies: one 540x640 observation frame + joint state + prompt → a
`(16, 8)` action chunk in 4 UniPC denoise steps. One adapter (backbone `cosmos3_policy`) serves
both — two `KNOWN_DECLARATIONS` entries differ only in size.

The adapter wraps **our patched cosmos-framework policy service** in-process
(`cosmos_framework.scripts.action_policy_server_robotwin.RobotwinPolicyService`): the request
goes through the *training-time* `ActionTransformPipeline`, which is the only way to keep
serve-time preprocessing byte-identical to train-time. The upstream release does not ship this
server — you need our patched checkout on the interpreter that hosts the model.

## Run it

```bash
pip install ./examples/cosmos3_policy
# from the patched cosmos-framework venv:
```

```python
from instinctflash import Runtime

runtime = Runtime.from_pretrained("nvidia/Cosmos3-Edge-Policy-DROID")
with runtime.episode(prompt="pick up the banana and place it in the bowl") as episode:
    action = episode.predict({"image": frame_hw3_uint8, "state": qpos8})   # -> (16, 8)
```

The serving config — `domain_name`, `action_dim`, `action_chunk_size`, image geometry, 4 steps,
guidance 1.0 — is **declared per checkpoint**, never guessed: those are the measured facts of
the published rows, and a wrong one silently skews preprocessing away from training. The policy
is stateless across control cycles (no KV survives a request), so episodes are per-call;
`reset()` reseeds the request stream.

## The two arms

| arm | H100 Edge | H100 Nano | note |
|:--|:--|:--|:--|
| NVIDIA stock (robolab, verbatim) | 310.5 ms | 482.3 ms | baseline |
| our pipeline (default) | 235.7 ms | 327.3 ms | optimization knobs OFF |
| + CUDA graphs (`IFL_COSMOS3_CUDA_GRAPHS=1`) | **185.8 ms (1.67x)** | **324.7 ms (1.49x)** | see caveat |

**The CUDA-graphs arm is an option, not a default.** It is `torch.compile(mode=
"reduce-overhead")` over the same weights, and inductor's cudagraph_trees **asserts on a prompt
change** — the speedup holds for single-prompt workloads only; multi-prompt serving must stay on
the pipeline arm. On Jetson Thor the graphs arm is measured *slower* (676 vs 660 ms Edge):
capture pays on launch-bound GPUs and this is not one. Tier: NUMERIC (vs our own eager:
<=1.6e-2 Edge / <=5e-2 Nano; null controls 0.0). Matched-input actions differ systematically
from the NVIDIA server (input-assembly/RNG alignment is an open task) — the latency comparison
stands; an equivalence claim does not.

## Reproduce the README H100 rows

```bash
IFL_COSMOS3_MODEL=edge CUDA_VISIBLE_DEVICES=<idle-gpu> examples/cosmos3_policy/reproduce_h100.sh
IFL_COSMOS3_MODEL=nano CUDA_VISIBLE_DEVICES=<idle-gpu> examples/cosmos3_policy/reproduce_h100.sh
```

Three arms, strictly serialized, byte-identical measure clients per arm.

## Attribution

Cosmos3 and its DROID policy checkpoints are NVIDIA's (OpenMDW-1.1). Nothing is vendored here
beyond two small measure clients and a guardrail-no-op launcher; the adapter imports the
patched checkout in-process.
