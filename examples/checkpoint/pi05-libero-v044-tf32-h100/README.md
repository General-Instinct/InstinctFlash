# pi0.5 LIBERO-v044 TF32 H100 operating point

This pointer package applies the declared pi0.5 TF32 + replay-safe static-KV operating point to the
LIBERO fine-tune at the pinned `8e174154` revision.  It exists separately from the `pi05_base`
latency package so a closed-loop certificate cannot accidentally name or load the wrong weights.

The paired runner is `examples/pi05_vla/run_tf32_closed_loop.py`.  It uses this declaration for the
treatment plan, prints `NUMERIC`, and runs the FP32 and TF32 arms in separate processes.
The v044 compatibility gate measured 185.960 ms FP32-highest versus 72.407 ms TF32 per
50-action chunk (2.568x), with maximum postprocessed action delta 8.988e-4 and deterministic
same-arm replay. This is numeric/latency evidence only; it is not closed-loop success evidence.

After the complete `10 x 50` run, normalize and certify it without supplying identities after the
fact:

```bash
python examples/pi05_vla/emit_tf32_closed_loop_outcomes.py \
  --input "$RUN/control_fp32.raw.jsonl" --output "$RUN/control_fp32.jsonl" \
  --arm control_fp32
python examples/pi05_vla/emit_tf32_closed_loop_outcomes.py \
  --input "$RUN/treatment_tf32.raw.jsonl" --output "$RUN/treatment_tf32.jsonl" \
  --arm treatment_tf32
python examples/pi05_vla/certify_tf32_closed_loop.py \
  --control "$RUN/control_fp32.jsonl" --treatment "$RUN/treatment_tf32.jsonl" \
  --run-manifest "$RUN/paired_run_manifest.json" --output "$RUN/certificate.json"
```

The certifier reads both operating-point hashes from the immutable run manifest and applies the
preregistered Tango paired one-sided 95% lower bound at the -5pp margin, plus the task-collapse
gate. It will not accept CLI hashes chosen after outcomes exist, a partial schedule, changed source
or checkpoint files, a different preregistration, or a run without passing pre-paired null controls.

Direct loads require explicit `tier_ceiling="numeric"` (serve CLI: `--runtime.tier_ceiling=numeric`). The package cannot raise the default BITEXACT ceiling.
