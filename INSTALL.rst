Install and use InstinctFlash
============================

The core requires Python 3.10 or newer. From the checkout root, install it in
a fresh CPU environment::

    python3 -m venv .venv-core
    .venv-core/bin/python -m pip install . uv==0.12.5
    .venv-core/bin/instinctflash models
    .venv-core/bin/instinctflash serve --help

The core installs Hub and YAML support. It does not install PyTorch, CUDA,
model weights or a vendor model stack. A wheel installation works from other
directories without setting PYTHONPATH. The ``models`` command reports known
checkpoint declarations and adapter registration; registration alone does not
verify that a model can run in the environment.

For inference on Thor, use the family's pinned bootstrap from the checkout root::

    source .venv-core/bin/activate
    python3 scripts/bootstrap_vendor.py plan pi05 --json
    python3 scripts/bootstrap_vendor.py install pi05 --python python3.12 --root ~/ifl-pi05 \
      --ptxas /usr/local/cuda/bin/ptxas
    source ~/ifl-pi05/activate.sh
    python -m pip check

The aliases are ``va``, ``vla4``, ``vla2``, ``pi05``, ``groot``, ``edge``,
``nano`` and ``dreamzero``. The output directory must be new. The bootstrap
resolves the pinned public vendor source, verifies the recorded compatibility
patches, creates a separate environment, and installs noneditable vendor, core
and adapter wheels. It writes dependency checks, source hashes and activation
settings under the output directory. It needs network access, Git, the selected
Python interpreter and ``uv``; supply an existing executable with ``--uv``.
``--ptxas`` checks that the selected CUDA assembler supports Thor's ``sm_110a``
target and saves both Triton compiler overrides in the activation. The measured
installation uses CUDA 13.2; some bundled Triton assemblers do not support Thor.

For RTX 4090, select the separate Linux/x86-64 profile explicitly::

    python3 scripts/bootstrap_vendor.py plan pi05 --target rtx4090 --json
    python3 scripts/bootstrap_vendor.py install pi05 --target rtx4090 \
      --python python3.12 --root ~/ifl-pi05-4090 --ptxas /usr/local/cuda/bin/ptxas
    source ~/ifl-pi05-4090/activate.sh
    python scripts/public_deploy.py doctor pi05 --target rtx4090

The same eight aliases are available. These profiles use x86-64 dependency
wheels and test the assembler against ``sm_89``; the Thor wheel repair and
Blackwell compiler override do not apply. ``release/rtx4090/deployment_profiles.json``
records each model's qualification status. A prepared profile or a successful
CPU doctor is not a completed GPU benchmark.

Large models need CPU backing memory on a single 24 GiB card. VA stages its
text encoder between resets. Cosmos and DreamZero can stream native layers;
DreamZero also transports complete per-layer KV caches. Native arithmetic,
checkpoint processors and history lengths remain intact, and transfer time is
included in prediction measurements. Backend statistics report actual residency.
The explicit SM89 FP8 path uses PyTorch/Triton projections; it does not load a
Thor-only native library. FP8 remains a separate numerical choice.

To share downloaded packages across separate compatible environments, use
``--cache-dir /path/to/cache --link-mode hardlink``. The cache and environment
must be on an executable filesystem that supports hard links. Model files can
live on a different filesystem; do not put an environment on a ``noexec`` mount.

If the required Python version is missing, ``uv python install 3.12`` (or
``3.13`` for Cosmos) installs it separately. Pass the path printed by
``uv python find 3.12`` as the bootstrap's ``--python`` argument.

Prepare the selected checkpoint and its exact tokenizer/processor assets::

    python scripts/prepare_auxiliary_assets.py prepare pi05 \
      --root ~/ifl-pi05-assets --include-primary
    source ~/ifl-pi05-assets/run.env
    python scripts/public_deploy.py doctor pi05

Use the selected model alias in both commands. Model downloads use your normal
Hugging Face authentication; accept any required upstream access conditions
first. Cosmos preparation includes the external Wan VAE. Keep the generated
asset activation alongside the vendor activation for subsequent sessions.

Install the published native libraries or build them using
`the backend guide <serving/README.rst>`_. The CPU core
environment above does not become a model environment by installing an adapter.
Use one environment per incompatible vendor stack; Edge and Nano can share the
Cosmos stack. All model adapters register automatically when installed.

For an already prepared, compatible vendor environment, you can install the
core and adapter directly, for example::

    /path/to/dreamzero-env/bin/python -m pip install '.[serve]' ./examples/dreamzero ./serving
    export DREAMZERO_ROOT=/path/to/dreamzero-repo

.. list-table:: Family installation requirements
   :header-rows: 1
   :widths: 18 26 10 46

   * - Family
     - Adapter directory
     - Thor Python
     - Model stack and source
   * - LingBot-VA
     - Built into the core
     - 3.12
     - Torch, Diffusers, Transformers, Safetensors and upstream Wan-VA;
       set LINGBOT_ROOT. requirements-serving.txt records the serving pins.
   * - pi05
     - examples/pi05_vla
     - 3.12
     - Torch and the matching LeRobot policy/processor stack, including
       access to the tokenizer named by the checkpoint's processor configuration.
   * - LingBot-VLA 4B
     - examples/lingbot_vla
     - 3.12
     - Torch, Torchvision, Transformers, Safetensors, YAML, LeRobot and
       the native checkout; set LINGBOT_VLA_ROOT.
   * - LingBot-VLA V2 6B
     - examples/lingbot_vla_v2
     - 3.12
     - Its matching Qwen3-VL/LeRobot stack and native checkout;
       set LINGBOT_VLA_V2_ROOT.
   * - GR00T N1.7
     - examples/groot_n17
     - 3.12
     - The matching Isaac-GR00T environment, including gr00t.policy;
       set GR00T_ROOT.
   * - DreamZero
     - examples/dreamzero
     - 3.12
     - The matching GEAR-Dreams environment, including Torch, OpenCV,
       Tianshou, openpi_client and native groot data modules;
       set DREAMZERO_ROOT. Native source compatibility is checked at load.
   * - Cosmos3 Edge and Nano
     - examples/cosmos3_policy
     - 3.13
     - The matching Cosmos Framework RoboLab policy environment,
       including its importable cosmos_framework package.

The Python column records the Thor vendor stacks; the core's Python 3.10 floor
does not imply that every vendor supports Python 3.10. The reviewed public
source pins, inference dependencies and compatibility patches live in
`release/vendor <release/vendor/README.rst>`_. The older
requirements-serving.txt is a LingBot-VA reference, not a universal Thor
environment.
The ``runtime`` extra supplies general Torch/Safetensors/NumPy support; it does
not supply every family's vendor stack. Install ``.[serve]`` for websocket
serving, and ``.[test]`` for CPU tests. Neither extra installs Torch.

The same ``Runtime`` API and ``serve`` command cover all eight models.
Run ``instinctflash models`` in the selected vendor environment to obtain
their Hub IDs and confirm adapter registration. For example, after installing
``.[serve]`` and the VLA4 adapter and setting ``LINGBOT_VLA_ROOT``::

    instinctflash serve robbyant/lingbot-vla-4b-posttrain-robotwin --serve.dry_run=true
    instinctflash serve robbyant/lingbot-vla-4b-posttrain-robotwin \
      --runtime.device=cuda:0 --serve.host=127.0.0.1 --serve.port=8000

The second command starts an OpenPI-compatible websocket server. Python callers
can instead use ``Runtime.predict`` directly, as shown below.

Inspect before loading
----------------------

Use a local checkpoint directory containing ``instinctflash.json`` and its
configuration, or a supported Hub model id::

    instinctflash describe /path/to/checkpoint --json
    instinctflash serve /path/to/checkpoint --serve.dry_run=true

Preflight reads metadata and reports the plan without downloading weights or
loading a model. Local preflight does not contact the Hub. Hub preflight may
fetch the declaration and adapter-required configuration metadata; DreamZero
needs config.json to determine the actual DiT geometry. When the declaration
comes from a Hub snapshot, its commit also pins those configuration files.
The Python metadata API can explicitly disable device probing::

    from instinctflash.runtime.facade import plan_declaration
    checkpoint, adapter, plan, device = plan_declaration(
        "/path/to/checkpoint", probe_device=False)
    print(plan.explain())

JSON and YAML configuration use the same fields. CLI overrides win::

    # serve.yaml
    serve:
      model: /path/to/checkpoint
      dry_run: true
    runtime:
      precision: native
      step_cache: checkpoint
    output:
      format: json

    instinctflash serve --config_path=serve.yaml --serve.dry_run=true

Unknown fields fail explicitly. ``--output.format=json`` returns structured
success and error records. ``--output.path=report.json`` writes a report file.

Load and predict
----------------

For example, with the LingBot-VLA 4B vendor environment and adapter installed::

    from instinctflash import Runtime

    with Runtime.from_pretrained(
        "/path/to/lingbot-vla-4b", device="cuda:0",
        precision="native", step_cache="checkpoint",
    ) as runtime:
        print(runtime.observation.describe())
        observation = runtime.observation.example()
        observation["prompt"] = "pick up the object"
        runtime.reset(prompt=observation["prompt"])
        result = runtime.predict(observation)
        print(result["action"].shape)
        print(runtime.backend_stats)

``example()`` creates synthetic inputs for an API smoke check. For the shipped
VLA4 RoboTwin declaration these are three uint8 RGB cameras of shape
``(480, 640, 3)`` under ``observation.images.cam_high``,
``observation.images.cam_left_wrist`` and ``observation.images.cam_right_wrist``,
plus a float32 ``observation.state`` of shape ``(14,)``. Replace them with the
current cameras and robot state for real use.

Other models have different inputs and action horizons. Inspect the resolved
contract and its ``runtime.observation_source``; adapter contracts are in
`LingBot-VA <instinctflash/adapters/lingbot_va.py>`_,
`pi05 <examples/pi05_vla/pi05_iwm/adapter.py>`_,
`VLA4 <examples/lingbot_vla/lingbot_vla_iwm/adapter.py>`_,
`VLA2 <examples/lingbot_vla_v2/lingbot_vla_v2_iwm/adapter.py>`_,
`GR00T <examples/groot_n17/groot_n17_iwm/adapter.py>`_,
`Cosmos <examples/cosmos3_policy/cosmos3_iwm/adapter.py>`_ and
`DreamZero <examples/dreamzero/dreamzero_iwm/adapter.py>`_.
LingBot-VA and DreamZero also require their episode-dependent camera history.
Reset once at an episode boundary, then retain the runtime across control cycles.

Construction can download checkpoint weights. Native in-process loading is
normally deferred until reset or predict; an FP8 engine can load during
construction. Use ``revision=<commit>`` for reproducible Hub selection.
Keep the loaded Runtime for repeated calls. The first prediction can include
calibration or graph capture; warm it with representative inputs before timing
control cycles, then reset before the real episode.

Native precision with a bitexact optimization ceiling is the default.
``precision="fp8"`` is an explicit request
that requires a supported engine, device and model geometry; it never silently
falls back. It defaults to a numeric ceiling and conflicts with an explicit
``tier_ceiling="bitexact"``. DreamZero can also select approximate dynamic prediction reuse::

    runtime = Runtime.from_pretrained(
        "/path/to/dreamzero", step_cache="dynamic", tier_ceiling="behavioral")

FP8 plus dynamic reuse requires both ``precision="fp8"`` and the explicit
behavioral ceiling. ``tier_ceiling="numeric"`` alone does not select FP8.
For native VLA2's graph path, pass ``precision="native",
tier_ceiling="numeric"`` explicitly: the default bitexact ceiling excludes
that optimization. A ceiling permits eligible optimizations; it does not
change the checkpoint schedule.
Dynamic reuse is currently integrated only for DreamZero.
``step_cache="checkpoint"`` restores the declaration and ignores legacy
DreamZero schedule environment variables. Omitting it preserves their legacy
selection behavior. Precision and step reuse are separate decisions.
DreamZero FP8 requires ``LOAD_TRT_ENGINE`` to be unset and
``ENABLE_TENSORRT`` disabled; its supported mask is the shipped 8-of-16 mask
or the explicit dynamic profile.

Omitting ``nfe`` preserves the checkpoint schedule. For the shipped RoboTwin
LingBot-VA that is 25 video and 50 action steps. An explicit
``nfe={"video": 2, "action": 4}, tier_ceiling="behavioral"`` selects a
different operating point. DreamZero's ``video_action=16`` describes solver
updates; its shipped mask computes eight denoiser steps, each with two CFG
branches. Dynamic reuse keeps the solver grid and varies the computed count.

``runtime.backend_stats`` returns a detached snapshot with status
``available``, ``not_loaded`` or ``unsupported``. It does not load the model
or contact a worker. ``runtime.execution_policy`` reports the selected schedule
and arithmetic policy. Where supported, realized dynamic-cache counts appear
inside ``stats`` after inference; planning estimates are not measured calls.
Use one active observation stream per runtime and reset between episodes.

Native engine assets
--------------------

The repository includes the CUDA source and build tools for FlashRT, FA2,
FMHA and shared BF16 fusion. Follow the complete
`Thor source build <serving/README.rst#jetson-thor-build>`_: first build FA2,
then build the remaining kernels against the pinned CUTLASS revision and
install the resulting platform wheel into the model environment::

    python -m pip install --force-reinstall --no-deps /path/to/native-output/wheels/flash_rt-*.whl
    python -m pip check

The build records source hashes, compiler commands, target architecture and
library hashes. A pure Python FlashRT wheel does not contain the compiled
accelerator. Build once for each compatible target/Python ABI; the Thor
``cp312`` wheel is for Python 3.12 on aarch64 and cannot be installed in Cosmos's
Python 3.13 environment. Cosmos Edge's NUMERIC recipe instead uses the separate
``native/libinstinctflash_bf16.so`` output through
``IFL_BF16_KERNEL_LIBRARY``. That library has a C interface and no CPython ABI.

The explicit reinstall replaces the bootstrap's source-only wheel even when
both carry the same package version. ``--no-deps`` preserves the family's
already checked dependency versions; the following ``pip check`` must pass.

Install the original checkpoint and auxiliary tokenizer/processor assets before
offline inference. Access to gated upstream assets requires accepting their
own license. The `reproduction guide <REPRODUCE.rst>`_ provides exact model
revisions, selected execution modes, paired captures and the real CLI/WebSocket
smoke test. A successful library import alone is not an inference test.

Check a built installation
-------------------------

From the checkout root, build the core and six adapter wheels, install them in
a new environment, and run the audit from outside the checkout::

    IFL_CHECKOUT="$(pwd)"
    IFL_AUDIT="$(mktemp -d /tmp/instinctflash-install.XXXXXX)"
    python3 -m venv "$IFL_AUDIT/venv"
    "$IFL_AUDIT/venv/bin/python" -m pip wheel --no-deps --wheel-dir "$IFL_AUDIT/wheels" \
      "$IFL_CHECKOUT" "$IFL_CHECKOUT/examples/pi05_vla" \
      "$IFL_CHECKOUT/examples/lingbot_vla" "$IFL_CHECKOUT/examples/lingbot_vla_v2" \
      "$IFL_CHECKOUT/examples/groot_n17" "$IFL_CHECKOUT/examples/cosmos3_policy" \
      "$IFL_CHECKOUT/examples/dreamzero"
    "$IFL_AUDIT/venv/bin/python" -m pip install "$IFL_AUDIT"/wheels/*.whl
    cd "$IFL_AUDIT"
    CUDA_VISIBLE_DEVICES="" "$IFL_AUDIT/venv/bin/python" -I \
      "$IFL_CHECKOUT/scripts/check_installed_package.py" \
      --require-torch-free --require-all-adapters --output installed-wheel-check.json

This CPU audit verifies
packaged assets, discovery, metadata-only planning and JSON/YAML CLI behavior.
It blocks Hub downloads and device probes and never runs model inference.
Package installation itself needs access to dependency wheels or a local wheelhouse.

Read the measured operating point
--------------------------------

The `public reproduction guide <REPRODUCE.rst>`_ records how to prepare exact
checkpoint revisions, Runtime arguments and optimizer environment options.
The `current results <eval/public_release_2026-09-15/results.rst>`_ report the
measured comparisons and their installed environments.
Some selected options are experimental; neither FP8 nor a numeric/behavioral
ceiling is a guarantee of faster execution or task quality.

The study compares eager native, default Runtime and explicitly selected Runtime
settings using the same requests and checkpoint schedules. Stateless cells use
five warmups and twenty measured requests, with episode resets and alternating
prompts. History cells use one warmup episode and six measured episodes of three
cycles; continuation latency covers the twelve later-cycle requests. pi05's
continuous 51-call queue check is separate from generation timing. Initial
loading/reset is reported separately; prompt setup deferred into predict is
inside the measured call. DreamZero eager setup also includes a full checkpoint
value audit; its recorded setup time is not a clean upstream startup benchmark.
Fixed-prompt steady-state timings are a different workload. Reduced LingBot-VA
steps and dynamic DreamZero reuse have separate
comparison groups and do not establish architecture-only speedups.

Native precision and a bitexact ceiling describe the selected checkpoint-relative
optimizations; they do not guarantee action equality against a separate reference
implementation. The current DreamZero native reference retains vendor encoder
compilation and uses an eager DiT. Set ``TORCHINDUCTOR_COMPILE_THREADS=1`` for its
Thor reproduction to limit compiler memory use. The
`earlier comparison <eval/user_e2e_2026-09-14/dreamzero_default_difference.json>`_
used a different compilation setup and retains its original action differences.

`release_selection.json <eval/user_e2e_2026-09-14/release_selection.json>`_
identifies the retained historical source and full-action evidence. Those
screens and the current API study do not measure robot task success.
