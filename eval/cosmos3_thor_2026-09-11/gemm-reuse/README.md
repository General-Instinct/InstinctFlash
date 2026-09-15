# Reusing the engine's BF16 GEMM on Thor

This screen calls the existing `serving/csrc/gemm/gemm_runner.cu` BF16 NN
implementation through a small C bridge. No FP8 execution is involved.
Synthetic inputs use the real Cosmos generation-token count (3094) and
Edge/Nano linear dimensions. `screen.json` records all eight shapes and the
compiled library hash.

Each timing uses CUDA Graph replay (five warmups, thirty replays), including a
repeated native measurement. Weight packing and algorithm tuning occur once
outside timing. All tested packed and engine outputs match native bytes for
these synthetic inputs; this does not establish model action equivalence.

The clearest simple candidate is Nano's 4096 → 12288 projection: prepacked
Torch GEMM takes 2.314 ms versus 2.521 ms for repeated native (~8% lower latency).
The first native measurement was 2.750 ms; use the repeat to avoid overstating
the gain. This motivates the separate `weight-layout` full-model experiment.

The tuned runner also improves some down projections, but tuning is not
uniformly beneficial: Edge's up projection regresses under graph replay.
An eager autotuning winner must therefore be checked in its actual execution
mode before integration. This screen changes no Runtime defaults.

`build.sh` builds the bridge with CUDA 13 for SM110a; set `GEMM_SOURCE` to the
directory containing the unchanged runner source and header. `screen.py ROOT`
expects `ROOT/bf16_gemm.so` and writes a fresh `ROOT/screen.json`. Run on an idle
Thor with the shared GPU lock. The benchmark is a synthetic screen, not a
closed-loop quality or complete performance regression certificate.
