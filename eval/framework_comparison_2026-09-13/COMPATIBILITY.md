# Thor framework setup

All GPU work holds `/tmp/thor_gpu.lock`. H100 measurements are excluded.

LeRobot uses pinned source `b6ec0060779550c0a157ae34feb89e0cf86012a8`. GR00T required `diffusers==0.38.0` and `dm-tree` in a benchmark-only import directory; the existing shared Python environment was not overwritten. Its first two failed attempts remain archived. Compiled pi05 uses the system CUDA ptxas, because the bundled assembler rejected SM110.

Omni uses pinned source `f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3`, vLLM 0.29.0 and a separate environment. Cosmos action policy uses the matching current Cosmos Framework imports. DreamZero additionally uses an isolated `omni-thor-compat-v1` source copy:

- The cached UMT5 tokenizer snapshot is supplied explicitly through `model_paths`.
- The generic Omni startup warmup sends two steps, while DreamZero only recognized a one-step dummy request. The isolated patch recognizes both dummy requests. Actual robot requests retain their configured schedule.
- Before KV memory admission, clean cached checkpoint pages receive Linux `POSIX_FADV_DONTNEED`. Thor shares CPU/GPU memory; otherwise CUDA's free-memory query caused admission to fail despite reclaimable file cache. This operation does not change checkpoint files or model math, and runs before latency samples.

The exact before/after source hashes and replacements are in the [compatibility manifest](omni_compatibility_manifest.json). Failed tokenizer, memory-admission and dummy-warmup attempts are retained. An Omni result using this copy must be labeled as requiring these startup compatibility fixes, rather than described as an unmodified installation.

LingBot-VA uses the released LeRobot conversion at `d1e1f93a84eaf9bca9880856fda800cc98cc8eaa`, with frozen modules from the native cached checkpoint. A CPU audit found all 839 converted tensors byte-identical to the corresponding native tensors. Native also retains two legacy `patch_embedding` tensors that the inspected forward paths do not reference. This weight audit does not prove inference equivalence. LeRobot drops the conditioning-frame actions from its first returned chunk (16 actions); later chunks return 32. The full two-frame generation and history protocol are retained and this return-layout difference is recorded.

The benchmark does not relabel upstream vendor servers as LeRobot or Omni. Registry support is retained in `framework_support_audit.json`.

The first pi05 NFE1 LeRobot probe did not merge the explicit Runtime schedule override into its reference constructor and actually used the checkpoint ten-step default. `excluded_receipts.json` excludes it from one-step selection; the corrected constructor applies the override before model construction and asserts both concrete policy and model step counts. The independent native Runtime NFE1 arm did receive its schedule override.

The VA harness supplies fixed recorded action history, normalized with the native checkpoint quantiles, through LeRobot's history buffers; the policy transformer and sampler are unchanged. History prompts stay fixed for each three-cycle episode. The earlier DreamZero eager v5 receipt switched one prompt mid-episode and is excluded from the aligned matrix; its corrected replacement uses the same episode-boundary prompt rule as Flash.
