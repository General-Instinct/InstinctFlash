Reproduce the Thor results
==========================

The benchmark, CUDA kernels and model frontends are included as source.
Each model runs in its own vendor environment. The fixed input archive contains
recorded RGB cameras and synthetic robot states; it loads without pickle.
Checkpoint commits, Runtime options, sample order and action shapes are explicit
in ``release/deployment_profiles.json`` and the prepared run directory.

Install a model environment
---------------------------

First complete the CPU core and ``uv`` setup in `INSTALL.rst <INSTALL.rst>`_.
The bootstrap requires Git, network access and the selected Python interpreter;
that guide also explains installing Python 3.12 or 3.13 with ``uv``.
From the checkout root, activate the core environment, then select one model
and an empty output directory::

    source .venv-core/bin/activate
    python3 scripts/bootstrap_vendor.py plan pi05 --json
    python3 scripts/bootstrap_vendor.py install pi05 --python python3.12 --root ~/ifl-pi05 \
      --ptxas /usr/local/cuda/bin/ptxas
    source ~/ifl-pi05/activate.sh

Other aliases are ``va``, ``vla4``, ``vla2``, ``groot``, ``edge``, ``nano`` and
``dreamzero``. Edge and Nano use Python 3.13; the other Thor environments use
Python 3.12. The bootstrap uses fixed public package/source versions, applies
the recorded compatibility patches in a new checkout, and installs noneditable
core/adapter wheels. It retains installation errors and receipts.

Vendor compatibility patches define the Thor baseline. In particular, the VLA
attention/MoE compatibility patches are disclosed separately from InstinctFlash
transformations. Dependency checks must pass before loading a model.

For the FP8 engine and Edge's shared BF16 fusion, install the published native
libraries or build them using `serving/README.rst <serving/README.rst>`_. Those builds retain
source hashes and compiler commands; they do not change checkpoint precision
until the caller explicitly selects an execution mode.

Prepare the checkpoint and inputs
--------------------------------

Prepare both the primary checkpoint and its auxiliary assets from the checkout
root, in the activated model environment::

    python scripts/prepare_auxiliary_assets.py prepare pi05 \
      --root ~/ifl-pi05-assets --include-primary
    source ~/ifl-pi05-assets/run.env

Replace ``pi05`` with the selected model alias. Accept the original model's
license/access conditions first; pi05 requires access to the gated PaliGemma
tokenizer through your Hugging Face login. Tokenizer/processor revisions and
the external Cosmos VAE are recorded in ``release/vendor/asset_profiles.json``.
The helper verifies each auxiliary file and records the pinned primary
checkpoint inventory in a new cache.

In a later shell, restore both activations before preparation, measurements or
serving; the benchmark bundle does not replace the vendor and asset environment::

    source ~/ifl-pi05/activate.sh
    source ~/ifl-pi05-assets/run.env

For Cosmos, the vendor activation also restores the prepared offline ``uvx``
tool used by its native loader. Keep both preparation directories available.

For example, after preparing pi05's vendor environment and native engine::

    python -I -m benchmarks.regression.reproduce prepare \
      --model pi05 --mode fp8 --output pi05-inputs

``--local-files-only`` reuses an already populated Hub cache without downloading.
The output contains the exact plan, comparison matrix and safe input archive.
Existing outputs are never overwritten. ``plan`` prints the same selection
without downloading weights, importing PyTorch or probing a GPU::

    python -I -m benchmarks.regression.reproduce plan --model pi05 --mode fp8

Use ``python -I`` for these installed benchmark commands, including when working
inside the checkout. It prevents checkout modules or ``PYTHONPATH`` from
shadowing the noneditable wheel that the runner validates.

RTX 4090 uses its own installation and measurement target::

    python -I -m benchmarks.regression.reproduce plan \
      --target rtx4090 --model pi05 --mode fp8
    python -I -m benchmarks.regression.reproduce prepare \
      --target rtx4090 --model pi05 --mode fp8 --output pi05-4090-inputs
    python -I -m benchmarks.regression.reproduce run \
      --prepared pi05-4090-inputs --output pi05-4090-results

Use the RTX 4090 bootstrap described in ``INSTALL.rst`` first. Preparation
binds the target, checkpoint and protocol into the bundle. Execution verifies
the selected card's model and SM89 capability; it does not reuse a Thor device
receipt. The paired native/default/selected requests retain the same warmups,
measurement counts, full observation shapes and history protocol. The native
reference declares any CPU residency needed to run the full checkpoint on a
24 GiB card. Those copies are included in latency, with actual peak allocated
and reserved CUDA memory recorded. The Thor selections below are not measured
RTX 4090 results; consult the target's qualified results before choosing a mode.

Explicit execution selections
-----------------------------

.. list-table:: Main README InstinctFlash selections
   :header-rows: 1

   * - Model alias
     - Mode
     - Policy
   * - va
     - fp8
     - Original 25 video / 50 action steps
   * - va
     - 2v4a-fp8
     - Separate 2 video / 4 action operating point
   * - vla4
     - fp8
     - Original action NFE10
   * - vla2
     - fp8
     - Original action NFE10
   * - pi05
     - fp8
     - LIBERO v044 checkpoint, action NFE10
   * - groot
     - native
     - Original action NFE4
   * - edge
     - numeric
     - Original BF16, UniPC4, CFG3; shared BF16 library required
   * - nano
     - numeric
     - Original BF16, UniPC4, CFG3
   * - dreamzero
     - dynamic-fp8
     - Explicit approximate step reuse with FP8

``--mode native`` is available for every model. It keeps checkpoint precision
and a BITEXACT transformation ceiling. The ceiling is permission for eligible
transformations; action agreement is measured against the separate eager
reference. FP8, NUMERIC and changed schedules have distinct selections and
never imply a task-quality certificate.

For Edge, bind the library produced by the native build during preparation::

    python -I -m benchmarks.regression.reproduce prepare \
      --model edge --mode numeric --output edge-inputs \
      --library IFL_BF16_KERNEL_LIBRARY=/path/to/native/libinstinctflash_bf16.so

For LingBot-VA, reproduce the full-schedule native reference and both
InstinctFlash operating points::

    python -I -m benchmarks.regression.reproduce prepare \
      --model va --mode fp8 --extra-mode 2v4a-fp8 --output va-inputs
    python -I -m benchmarks.regression.reproduce run \
      --prepared va-inputs --output va-results

This report compares both candidates against original 25V/50A eager execution
and labels 2V/4A as a different sampling policy. The README's 2V/4A row uses an
additional native 2V/4A measurement for its same-schedule ratio; run the
`native 2V/4A recipe <eval/va_native_2v4a_2026-09-15/README.rst>`_
from the current source installation to reproduce that cell.

For DreamZero, limit concurrent compiler workers before running or serving::

    export TORCHINDUCTOR_COMPILE_THREADS=1

The Thor qualification uses this setting with the compiler selected by
``--ptxas``. Leave sufficient shared CPU/GPU memory available: unrelated
environments or caches stored in ``/dev/shm`` consume the same memory budget.
This controls compilation concurrency; it does not change diffusion steps.

Measure and inspect
-------------------

Run on an otherwise idle Jetson Thor, using the prepared model environment::

    python -I -m benchmarks.regression.reproduce run \
      --prepared pi05-inputs --output pi05-results

Each arm starts in a fresh process. The run compares eager native, default
Runtime and the explicit selected Runtime mode. Changed schedules add a
separate operating-point comparison. Inherited optimizer overrides are cleared
before applying the recorded options; run-time model resolution is offline.

Stateless models use five warmups and twenty measured generations. LingBot-VA
and DreamZero retain history across three-cycle episodes: one warmup episode
and six measured episodes. Their primary timing uses continuation cycles.
pi05's separate 51-call queue check includes actual queue drains; queue-hit
latency is not reported as model generation speed.

``validated_report/report.json`` and ``report.csv`` include prediction latency,
matched ratios, full-action differences and validation status. Raw action arrays,
request hashes, source identities, initial-load time and failures remain in the
run directory. Recheck an existing run without using a GPU::

    python -I -m benchmarks.regression.reproduce report \
      --run pi05-results --output pi05-report-recheck

Absolute latency depends on clocks, temperature, background load and the
declared request protocol. Compare the same checkpoint, inputs, steps, precision
and cache policy. README historical LeRobot/vLLM-Omni cells use separately pinned
protocols; they are not matched architecture-only ratios.

The separate `framework comparison guide <benchmarks/regression/FRAMEWORK_COMPARISON.rst>`_ records
those exact upstream commits, patches, checkpoints and measurement commands.

Verify the serving pipeline
---------------------------

After the paired run, test the actual installed CLI and wire protocol::

    python -I -m benchmarks.regression.serve_smoke \
      --prepared pi05-inputs --output pi05-serving

This starts one loopback CLI server, checks its advertised precision and step
schedule, sends two episodes with three requests each, validates finite action
shapes and preserves history/feedback. It stops the process it started and saves
the server log, actions and receipt. These six network requests include startup
effects and are a serving smoke test, not a substitute for the paired benchmark
or robot task evaluation.

Native smoke requests use a serving seed of 9173 unless overridden. FP8 serving
does not support the constructor seed option, so its transport smoke is
unseeded; explicitly supplying ``--seed`` for FP8 fails. The paired benchmark
seeds each model call independently and archives the resulting action arrays.

To keep the same tested selection running for a robot client, restore the vendor
and asset activations above, then reuse the successful smoke's configuration
and its prepared cell environment::

    python -I - pi05-inputs pi05-serving <<'PY'
    import json
    import os
    from pathlib import Path
    import sys
    from benchmarks.regression.reproduce import child_environment, sha, validate_bundle
    from benchmarks.regression.serve_smoke import selected_cell

    prepared, smoke = (Path(value).resolve() for value in sys.argv[1:])
    plan, preparation = validate_bundle(prepared)
    checked = json.loads((smoke / "receipt.json").read_text())
    assert checked["status"] == "passed"
    assert checked["plan_sha256"] == sha(prepared / "plan.json")
    assert checked["config_sha256"] == sha(smoke / "serve.json")
    cell = selected_cell(plan, checked["cell_id"])
    command = [sys.executable, "-I", "-m", "instinctflash.cli", "serve",
               f"--config_path={smoke / 'serve.json'}",
               "--serve.host=0.0.0.0", "--serve.port=8000"]
    os.execve(sys.executable, command, child_environment(plan, cell, preparation))
    PY

Replace the two directory arguments for another prepared model. This preserves
the selected runtime flags and applies its exact optimizer environment, including
the bound Edge BF16 library. A bare ``instinctflash serve`` command does not read
the benchmark profile or those environment settings. The server keeps running
until interrupted; its checkpoint, cache and library paths must remain available.

For your robot, inspect ``runtime.observation.describe()`` and replace the
recorded fixture with current cameras and robot state. Reset at episode
boundaries and report actions actually executed when a controller changes them.
See `INSTALL.rst <INSTALL.rst>`_ for Python and OpenPI-compatible client examples.

Task quality and historical experiments
--------------------------------------

Speed, action differences and closed-loop success are separate results. The
Edge/Nano UniPC4/CFG3 two-task SCREEN does not meet the selected one-sided 95%
confidence / five-percentage-point non-inferiority gate. Experimental SDE1
timings use different checkpoints or sampling policies and remain unqualified.
Use the paired simulator evaluation tooling for checkpoint-specific task tests;
never infer task success from a fast latency or a finite action array.

The `experimental SDE1 package <examples/cosmos3_sde1/README.rst>`_ provides the
separate Edge student overlays and original-weight Nano recipe behind the
earlier speed records. It pins their older base checkpoints and sampling
contracts, and does not promote them to the UniPC4 quality profile.
