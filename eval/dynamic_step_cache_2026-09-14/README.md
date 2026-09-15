# Dynamic step-cache investigation

Research completed on 2026-09-14. No model inference, GPU worker, new latency
result or task-quality admission. The shared public API is a proposal, not an
installed backend.

- [Current integration guide](../../docs/dynamic-step-cache.md): public policy,
  controller/adapter boundaries, native/FP8 qualification and supported families.
- [Source audit](source_audit.md), [source bindings](source_audit.json): pinned
  DreamZero and Omni behavior, request lifecycle and limits of older evidence.
- [CPU oracle](cpu_probe.py), [results](cpu_probe.json): 2,949 original-function
  decision/countdown comparisons passed, including 160 random generations and
  192 random histories. It does not test model outputs or task success.

Reproduce the CPU probe with the audited sources available:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python eval/dynamic_step_cache_2026-09-14/cpu_probe.py --output /tmp/step-cache-probe.json
```

Source paths can be overridden with `--native-source`, `--omni-state` and
`--omni-config`; hashes must match the audited versions. No model packages are
imported. Torch uses CPU tensors only, and CUDA initialization is checked absent.

The existing adapter, FP8-builder and owned-loading CPU regressions also passed
all 21 tests. An existing schedule-refusal test was repaired to reach declaration
validation without requiring a real GPU. Production changes in this investigation
correct comments, notes and logging; cache selection and inference math are unchanged.

Next implementation: a generation-owned controller and explicit Runtime policy,
DreamZero native parity, a fresh Thor native/FP8 x fixed/dynamic ablation, then
checkpoint-specific quality gates and additional family adapters.
