Reproduce the foreign-framework columns
=======================================

The portable runner reproduces the six historically measured LeRobot and
vLLM-Omni cells. It fixes the public source commits, checkpoint revisions,
recorded inputs, precision, schedules, warmups and sample selection. Install
and run each framework in its own environment on Jetson Thor with Python 3.12.
Native PyTorch and InstinctFlash use the separate ``reproduce`` command.

From this source checkout, inspect the plan and create NEW environments::

  python scripts/bootstrap_framework_compare.py plan --framework lerobot --root /new/lerobot --python /path/to/python3.12
  python scripts/bootstrap_framework_compare.py install --framework lerobot --root /new/lerobot --python /path/to/python3.12
  python scripts/bootstrap_framework_compare.py install --framework vllm-omni --root /new/omni --python /path/to/python3.12 --allow-startup-patch

``--uv`` selects an installed uv executable; ``--cache-dir`` selects a package
cache. ``--env-dir`` can place the fresh environment in tmpfs while retaining
source clones, installation logs and failure receipts under ``--root``.
There is no site-packages inheritance. The source pins and direct public
Torch/vLLM wheel URLs and hashes are in ``fixtures/frameworks/catalog.json``
and ``sources.json``. The public cuSPARSELt wheels (0.8.0 for LeRobot, 0.8.1 for
Omni) need the published
metadata-only AArch64 tag repair; the installer preserves all native bytes
and requires both pip and uv package checks. An existing audited artifact can
be selected with ``--repaired-wheel`` and ``--repair-receipt``.
Full clean-environment/GPU qualification remains a
separate gate; installing packages alone does not establish a table result.

Prepare exact model assets, then run one cell at a time::

  /new/lerobot/env/bin/python benchmarks/regression/framework_compare.py prepare lerobot-pi05 --output /new/pi05-inputs
  python benchmarks/regression/framework_compare.py run --prepared /new/pi05-inputs --output /new/pi05-run --python /new/lerobot/env/bin/python --ptxas /usr/local/cuda/bin/ptxas
  /new/omni/env/bin/python benchmarks/regression/framework_compare.py prepare vllm-omni-edge --output /new/edge-inputs
  python benchmarks/regression/framework_compare.py run --prepared /new/edge-inputs --output /new/edge-run --python /new/omni/env/bin/python --allow-startup-patch --ptxas /usr/local/cuda/bin/ptxas

The other cell names are ``lerobot-groot``, ``lerobot-va``,
``vllm-omni-nano`` and ``vllm-omni-dreamzero``. ``list`` displays the complete
catalog; ``plan CELL`` does not import Torch or create files. ``prepare``
accepts ``--cache-dir`` and ``--local-files-only``. It hashes every resolved
asset and checks all 839 mapped LingBot-VA tensors against their native
counterparts before preparing VA. Model downloads remain subject to their
original access and license terms.

PaliGemma and Cosmos-Reason2 processor files are separately hash-pinned and
mapped into a new owned auxiliary cache, without changing existing Hub refs.
DreamZero downloads only its four bound UMT5 tokenizer files; UMT5 model
weights are not required by this benchmark.

``run`` uses a fresh subprocess, a nonblocking Thor GPU lock and a finite
timeout (``--timeout-seconds``, default 3600). Results require finite action
arrays, exact request/order/seed bindings, a clean policy close and actual
process exit zero. Raw actions, queued pi05 actions, logs, versions, source
hashes and all per-call times are retained. Re-run the CPU report check with::

  python benchmarks/regression/framework_compare.py report --prepared /new/pi05-inputs --output /new/pi05-run

Interpretation
--------------

LeRobot pi05 uses compiled NFE1; its reported speed is not a same-schedule
comparison to InstinctFlash NFE10. LeRobot GR00T uses NFE4. LeRobot VA uses
2V/4A, fixed recorded feedback and the upstream first-chunk return layout.
Omni Edge/Nano use compiled UniPC4/CFG3. Omni DreamZero uses the pinned
16-step/CFG5 upstream step cache. VA and DreamZero report continuation
cycles 1 and 2 after one warmup episode: 20 selected samples from 11 episodes
of three calls. Other cells use 10 warmups and 30 measured calls. Reports
retain both selected and all-measured medians.

Omni requires two explicit historical startup fixes: recognize the two-step
dummy warmup and advise eviction of clean checkpoint pages before unified
memory admission. The portable variant binds the prepared physical checkpoint
directory in ``VLLM_OMNI_CHECKPOINT_PATHS``, separately from the small processor
cache. Without that explicit binding, it selects ``HF_HUB_CACHE``, then
``HUGGINGFACE_HUB_CACHE``, then the native default cache. Only clean files of
at least 64 MiB receive eviction advice; no files are changed. This runs
before measurement. The original historical patch and the portable variant
have separately published source hashes. The companion Cosmos dependency
declaration is adjusted to Omni's Transformers version for its pose,
transforms and UniPC imports. The default utility dependencies also explicitly
include historical ``iopath==0.1.10`` and ``portalocker==4.3.0``: upstream puts
iopath in a training extra, but the first inference request imports it through
``ActionTransformPipeline``. Bootstrap checks that exact symbol in a separate
CPU process with network, weight-file access and PyTorch initialization blocked.
No Cosmos implementation is modified.

Missing results remain missing. In particular, the pinned Omni registry has
a GR00T N1.7 entry, but the old matrix contains no qualified measurement for
it. That cell is **not qualified**, rather than proof of absent registry
support. Latency screens never establish closed-loop task accuracy, and
hardware/software changes can change the measured milliseconds.

Primary upstream sources: https://github.com/huggingface/lerobot,
https://github.com/vllm-project/vllm-omni,
https://github.com/NVIDIA/cosmos-framework. The manifest fixes exact commits;
current upstream defaults are not substituted for them.
