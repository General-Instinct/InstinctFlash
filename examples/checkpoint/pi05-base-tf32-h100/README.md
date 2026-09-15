# pi0.5 TF32 H100 operating point

This pointer-only package keeps the upstream `lerobot/pi05_base` weights and declares a different
execution contract: TF32 FP32 GEMMs plus replay-safe static-KV CUDA Graphs on SM90. This package requires explicit NUMERIC permission; selecting it alone does not
raise the default BITEXACT ceiling.

`instinctflash plan examples/checkpoint/pi05-base-tf32-h100 --tier-ceiling numeric` reports the
`pi05_tf32_numeric` pass as `APPLY [NUMERIC]` and the whole plan as NUMERIC. The ordinary pi05
declaration does not carry `execution.pi05_tf32_numeric`, so its plan remains BITEXACT.

Real H100 evidence is in `examples/pi05_vla/tf32_static_h100_results.json`: 261.37 ms FP32 eager,
165.96 ms TF32 eager, and 66.48 ms TF32 plus static capture for a full 50-action chunk. Four paired
input cases (including valid-token masks 48 -> 31 -> 48 -> 19) were deterministic; static replay was
bit-exact against TF32 eager, while the worst action delta against FP32 was 0.002482.

Closed-loop non-inferiority is not yet certified. The preregistered protocol and runnable outcome
comparator are next to the pi plugin. Do not interpret the tensor-delta gate as task-success evidence.

The gated tokenizer licence is accepted and the v044 checkpoint is local. The dedicated v044
pointer package and resumable runner are in `examples/checkpoint/pi05-libero-v044-tf32-h100` and
`examples/pi05_vla/run_tf32_closed_loop.py`. The real 500-pair run has still not been executed.
`emit_tf32_closed_loop_outcomes.py` converts each evaluator arm into exact-pairing JSONL;
`certify_tf32_closed_loop.py` then enforces the preregistered pair count and declared margin.
Because PyTorch's matmul precision is process-global, the adapter refuses to co-host FP32 and TF32
pi0.5 runtimes in one interpreter; use separate worker processes for mixed operating points.
