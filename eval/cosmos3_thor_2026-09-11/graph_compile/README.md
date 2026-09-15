# Cosmos Edge selective CUDA Graph experiment on Thor

Pinned upstream `2b6c9a7`, compiled BF16 attention experiment, four steps,
CFG 3, no diffusion cache. These results are within the numerical experimental
lane; they do not qualify a BITEXACT transformation of original upstream.

| Configuration | p50 ms | Max action delta vs dynamic reference | Graph replays |
|---|---:|---:|---:|
| Dynamic compile, no graphs (static_compile reference A) | 1310.48 | 0 | 0 |
| MoT graphs, padded query | 1513.48 | 0.15982 | 3472 |
| MoT graphs, unpadded query | 1316.27 | 0.20104 | 3472 |

Both graph runs produced finite 16 × 32 × 8 actions. Neither improves latency;
neither preserves the reference action bytes. Production defaults and accepted
regression baselines remain unchanged.

Capturing all compiled heads first failed: `_grouped_mm` in action encoding
does not support this capture. The working experiment compiles VFM heads without
graphs and enables graphs only for the MoT blocks. The first attempt to copy
configuration incorrectly treated an OmegaConf DictConfig as an attrs class;
the corrected copy preserves the original configuration.

Padding query tokens from 3094 to 3328 costs about 197 ms in this experiment.
Removing that padding recovers the cost but provides no gain over no graphs.
The attention Python-call count falls to 224 because it counts capture/warmup
dispatches, not GPU replay execution. The separate replay counter confirms
3472 actual CUDA Graph replays in each completed arm.

The scripts preserve each attempt, including failed configurations, and
`receipts/` contains errors, successful reports, audit metadata and full action
arrays. `run.py ROOT edge` runs the final unpadded candidate using the explicit
frozen environment paths. Compare against the sibling `static_compile/receipts`
dynamic A reference with `python compare.py /path/to/graph_compile/receipts`.
The comparator checks hashes, finite full actions, matched protocols, library
identity and actual graph replay coverage. No closed-loop task-quality
evaluation is claimed.
