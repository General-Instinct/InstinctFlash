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

Jetson Thor
-----------

For inference on Thor, use the family's pinned bootstrap from the checkout root::

    source .venv-core/bin/activate
    python3 scripts/bootstrap_vendor.py plan pi05 --target jetson_thor --json
    python3 scripts/bootstrap_vendor.py install pi05 --target jetson_thor \
      --python python3.12 --root ~/ifl-pi05 --ptxas /usr/local/cuda/bin/ptxas
    source ~/ifl-pi05/activate.sh
    python -m pip check

The aliases are ``va``, ``vla4``, ``vla2``, ``pi05``, ``groot``, ``edge``,
``nano`` and ``dreamzero``. The output directory must be new. The bootstrap
resolves the pinned public vendor source, verifies the recorded compatibility
patches, creates a separate environment, and installs noneditable vendor, core
and adapter wheels. It writes dependency checks, source hashes and activation
settings under the output directory. It needs network access, Git, the selected
Python interpreter and ``uv``; supply an existing executable with ``--uv``.
A failed required CPU doctor returns a nonzero exit status while preserving the
complete installation receipt and doctor log. ``--defer-doctor`` explicitly
leaves that check pending; it does not certify an inference environment.
``--ptxas`` checks that the selected CUDA assembler supports Thor's ``sm_110a``
target and saves both Triton compiler overrides in the activation. The measured
installation uses CUDA 13.2; some bundled Triton assemblers do not support Thor.

RTX 4090
--------

The pinned ``cu130`` environments require a CUDA 13-compatible NVIDIA driver
(`R580 or newer <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>`_).
The recorded RTX 4090 qualification used driver 610.43.02 with the CUDA 12.8
``ptxas``. The ``--ptxas`` option selects the
kernel assembler separately from PyTorch's CUDA runtime.

For RTX 4090, select the separate Linux/x86-64 profile explicitly::

    source .venv-core/bin/activate
    python3 scripts/bootstrap_vendor.py plan pi05 --target rtx4090 --json
    python3 scripts/bootstrap_vendor.py install pi05 --target rtx4090 \
      --python python3.12 --root ~/ifl-pi05-4090 --ptxas /usr/local/cuda/bin/ptxas
    source ~/ifl-pi05-4090/activate.sh
    python scripts/public_deploy.py doctor pi05 --target rtx4090

The same eight aliases are available. Edge and Nano use Python 3.13; VA, VLA4,
VLA2, pi05, GR00T and DreamZero use Python 3.12. These profiles use x86-64 dependency
wheels and test the assembler against ``sm_89``; the Thor wheel repair and
Blackwell compiler override do not apply. ``release/rtx4090/deployment_profiles.json``
records each model's qualification status. A prepared profile or a successful
CPU doctor is not a completed GPU benchmark.

Large models need CPU backing memory on a single 24 GiB card. VA stages its
text encoder between resets. Cosmos and DreamZero can stream native layers;
DreamZero also transports complete per-layer KV caches. Native arithmetic,
checkpoint processors and history lengths remain intact. Transfers during
prediction are included in its latency; VA's text-encoder staging occurs during
reset, outside the prediction timer. Backend statistics report actual residency.
The explicit SM89 FP8 path uses PyTorch/Triton projections. FP8 remains a
separate numerical choice.

The current source was qualified on one RTX 4090 node (a 24 GiB card, a 143 GB container
memory limit and 71 GB of ``/dev/shm``) from clean installs; the
`RTX 4090 results <release/rtx4090/results/results.rst>`_ list each family's recorded
modes. DreamZero's pinned model files (81 GB) exceed that node's model storage, so
DreamZero was not tested there. A capacity limit of one node does not establish that a
model is unsupported on RTX 4090 with more storage or host memory.

RTX 5090
--------

Select the separate Linux/x86-64 SM120 profile::

    source .venv-core/bin/activate
    python3 scripts/bootstrap_vendor.py plan pi05 --target rtx5090 --json
    python3 scripts/bootstrap_vendor.py install pi05 --target rtx5090 \
      --python python3.12 --root ~/ifl-pi05-5090 --ptxas /usr/local/cuda/bin/ptxas
    source ~/ifl-pi05-5090/activate.sh
    python scripts/public_deploy.py doctor pi05 --target rtx5090

Point ``--ptxas`` at an assembler supporting ``sm_120`` and PTX 8.7; the path
above is a typical toolkit location; Triton 3.6's bundled ``ptxas-blackwell``
12.9 is one such assembler. Its version is independent of the pinned PyTorch
``cu130`` runtime, which requires a CUDA 13-compatible NVIDIA driver. Activation
sets both Triton assembler overrides. Thor and SM89 compiler settings are separate.

All eight aliases and 23 explicit execution modes have RTX 5090 profiles.
Edge/Nano use Python 3.13 and Torch 2.10; the other families use Python 3.12 and
Torch 2.11. Keep the profile's original dependency pins. Native precision and
FP8 remain separate choices; FP8 does not imply faster inference or task-quality
equivalence. Large checkpoints may also require substantial host RAM.

RTX 5090 profiles are pending re-qualification of the current source on an RTX 5090;
``release/rtx5090/deployment_profiles.json`` records no current GPU measurement. The
README presents Thor results only. Continue with the
`RTX 5090 reproduction workflow <REPRODUCE.rst#rtx-5090>`_ for original assets,
paired predictions and WebSocket checks.

Model environments and assets
-----------------------------

To share downloaded packages across separate compatible environments, use
``--cache-dir /path/to/cache --link-mode hardlink``. The cache and environment
must be on an executable filesystem that supports hard links. Model files can
live on a different filesystem; do not put an environment on a ``noexec`` mount.

For a transported dependency cache, ``--dependency-wheelhouse /path/to/cache``
accepts a manifest bound to the selected recipe. The bootstrap checks every
wheel's size, SHA256 and package metadata, including the original digests of
explicit URL pins, then resolves dependencies without an index. The admission
receipt records the cache used. This option covers dependency wheels; vendor
source checkout and separately declared tools still follow their own setup.
When installing several families from transported caches, add
``--dependency-artifact-cache /path/to/shared-archives``. Identical verified
wheels then use the same file URLs, so ``uv`` can reuse their extracted files.
Combine this with the executable shared ``--cache-dir`` and ``--link-mode hardlink``.

Cosmos also uses a separate pinned HF command-line tool. Add
``--native-tool-wheelhouse /path/to/hf-tool-cache`` to prepare it offline.
Its ``instinctflash.native_tool_wheelhouse.v1`` manifest binds the tool catalog,
constraint hashes and all 23 original wheels; it is separate from the model's
dependency wheelhouse. The bootstrap verifies file hashes and package metadata,
keeps owned copies of these small archives, and checks the unchanged native
``uvx --with click hf@1.16.4`` command offline. The transport directory can then
be removed. Without this option, initial tool preparation uses the public index.

To reuse an existing vendor Git checkout or a self-contained Git bundle, add
``--vendor-source /path/to/prior/vendor`` or ``--vendor-source /path/to/vendor.bundle``.
The bootstrap fetches the profile's exact commit into its new destination and
runs the same source and patch checks. Local working-tree edits are excluded;
the input repository is unchanged. The receipt retains the original repository
URL and revision and records the local transport separately, including a bundle's
SHA256. A missing commit or incomplete bundle fails without a network fallback.
Without this option, the bootstrap fetches from the pinned public repository.

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
For RTX 4090 or RTX 5090, also pass ``--target rtx4090`` or ``--target rtx5090``
to the doctor command; asset preparation uses the same original checkpoints.

The Thor FP8 engines and Edge's fused BF16 linear are Python code and Triton
kernels compiled on first use; no native library is built or installed. The CPU core
environment above does not become a model environment by installing an adapter.
Use one environment per incompatible vendor stack; Edge and Nano can share the
Cosmos stack. All model adapters register automatically when installed.

For an already prepared, compatible vendor environment, you can install the
core and adapter directly, for example::

    /path/to/dreamzero-env/bin/python -m pip install '.[serve]' ./examples/dreamzero
    export DREAMZERO_ROOT=/path/to/dreamzero-repo

.. list-table:: Family installation requirements
   :header-rows: 1
   :widths: 18 26 10 46

   * - Family
     - Adapter directory
     - Python
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

The Python versions apply to the separately pinned Thor, RTX 4090 and RTX 5090 vendor
stacks; the core's Python 3.10 floor does not imply that every vendor supports
Python 3.10. The reviewed public source pins, inference dependencies and
compatibility patches live in
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
For the shipped RoboTwin VA checkpoint, send one current camera frame on the
first call, four newly observed frames on the second, and eight on later calls.
Each frame is a camera dictionary under ``obs``. DreamZero DROID uses one frame
per camera initially and four per camera on each continuation call. Use frames
observed while the previous action chunk executed; repeating a frame to fill
the history changes the model input.

One Runtime has one active episode. ``runtime.episode(prompt=...)`` offers a
context-managed handle to the same backend state; it does not isolate concurrent
episodes. Finish one episode before starting another and serialize calls.

For VA, pass the previous chunk actually executed with
``runtime.predict(observation, executed_action=previous_executed_chunk)`` when
it differs from the returned chunk. Omit it when the returned chunk was executed
unchanged. DreamZero's native wrapper rejects executed-action overrides.

Construction can download checkpoint weights. Native in-process loading is
normally deferred until reset or predict; an FP8 engine can load during
construction. Use ``revision=<commit>`` for reproducible Hub selection.
Keep the loaded Runtime for repeated calls. The first prediction can include
kernel compilation or graph capture; warm it with representative inputs before
timing control cycles, then reset before the real episode.

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

Connect a WebSocket client
--------------------------

The ``serve`` command uses OpenPI's msgpack/numpy wire format. A client can use
the dependencies already supplied by ``.[serve]``. After starting the loopback
server above, this generator sends one episode's iterable of observations and
yields the returned action chunks::

    from websockets.sync.client import connect
    from instinctflash.serving import msgpack_numpy

    def receive(connection):
        reply = connection.recv(timeout=120)
        if isinstance(reply, str):
            raise RuntimeError(reply)
        return msgpack_numpy.unpackb(reply)

    def stream_episode(observations, instruction):
        with connect("ws://127.0.0.1:8000", compression=None, max_size=None) as connection:
            metadata = receive(connection)  # The first frame is server metadata.
            print(metadata["observation"])
            connection.send(msgpack_numpy.packb({"reset": True, "prompt": instruction}))
            receive(connection)  # Reset acknowledgment; no action is returned.
            for observation in observations:
                observation = dict(observation, prompt=instruction)
                connection.send(msgpack_numpy.packb(observation))
                yield receive(connection)["action"]

Supply observations from the current cameras and robot state using the advertised
contract; execute each yielded chunk before collecting the next observation.
Keep the connection for subsequent control cycles. Send the reset message at
every episode boundary, even if the prompt is unchanged, and after reconnecting.
OpenPI client 0.1.1's ``reset()`` sends no message; with that client use
``client.infer({"reset": True, "prompt": instruction})`` explicitly.
A changed prompt also starts a new episode, but an unchanged prompt does not.

The Python history requirements above apply to each wire observation too.
For VA feedback, put ``executed_action`` in the observation map; the server
passes it to ``Runtime.predict`` as the keyword argument described above.
Do not send this field for DreamZero. All connections to one server share its
single runtime episode; use one active robot observation stream per server.
Adjust the receive timeout for the selected model's startup and prediction time.

Engine and model assets
-----------------------

The Thor FP8 engines (``instinctflash/engines``) need only the model
environment's PyTorch (with ``torch._scaled_mm`` for their FP8 GEMMs) and
Triton. Activation scales are computed per token as the data flows, so there is
no calibration step. Their Triton kernels are compiled for the device on first
use and cached by Triton; the first call of a new geometry therefore includes
compilation and CUDA graph capture. On Thor, the attention launches that opt
in (pi05 vision/prefix, LingBot-VLA prefix, GR00T VLSA, LingBot-VA self/cross)
and Edge's fused BF16 linear are compiled for ``sm100`` so their ``tl.dot`` uses
tcgen05; the other kernels use the default SM110 target, and the FP8 GEMMs are
``torch._scaled_mm``. Set ``TRITON_PTXAS_PATH`` (and
``TRITON_PTXAS_BLACKWELL_PATH``) to the CUDA 13 ``ptxas`` as the bootstrap's
``--ptxas`` option does.

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
The `README results <README.md#results>`_ were measured on September 15, 2026.
Their LingBot-VA, LingBot-VLA and pi05 FP8 cells and Edge's NUMERIC cell used
the earlier Thor engines, which InstinctFlash's own engines and Triton kernels
have since replaced; those cells have not been re-measured with the current
source.
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
implementation. The Thor DreamZero native reference retains vendor encoder
compilation and uses an eager DiT. Set ``TORCHINDUCTOR_COMPILE_THREADS=1`` for its
Thor reproduction to limit compiler memory use. The
4090 reference uses eager components with declared CPU weight/KV residency. An
earlier DreamZero comparison used a different compilation setup and retains its
original action differences.

Neither the earlier action screens nor the current API study measures robot task
success.
