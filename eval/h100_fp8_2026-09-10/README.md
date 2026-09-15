# H100 FP8 Runtime sweep

This study compares native and explicit FP8 execution through the same public Runtime interface. H100 uses an E4M3 attention-projection executor, distinct from the Thor fused kernels. FP8 changes arithmetic; it is not a quality-preserving default or a promise of higher speed.

The measured source is frozen under `source-v1` in the raw campaign directory. Nine pairs cover the eight model/checkpoint rows and LingBot-VA's separate 2V/4A operating point. The summary is published only after all pairs pass receipt, source, input, schedule, finite-output and timing checks.

See [measurement protocol](PROTOCOL.md). `benchmark.py` runs one arm; `run_lane.py` schedules sequential pairs on separate GPUs; `summarize.py` refuses incomplete or mismatched pairs. The raw campaign is `/home/ubuntu/ifl_eval/h100_fp8_20260910` on the measurement host. Its `lane-*.json` files track execution, including failures. Setup smoke attempts never supply table cells.

Use `Runtime.from_pretrained(checkpoint, precision="fp8")` or the existing `--fp8` CLI option. `precision="native"` remains the default. Only the listed checkpoint revisions and workloads are measured; H100 simulator quality is not established. Thor simulator scores do not certify this different FP8 executor.
