Public Thor inference environments
==================================

From the public checkout, create one isolated environment per vendor family::

    python3 scripts/bootstrap_vendor.py plan vla4 --json
    python3 scripts/bootstrap_vendor.py install vla4 --python python3.12 --root ./deploy-vla4
    source ./deploy-vla4/activate.sh
    python scripts/prepare_auxiliary_assets.py prepare vla4 --root ./assets-vla4 --include-primary
    source ./assets-vla4/run.env
    python scripts/public_deploy.py doctor vla4

Targets are pi05, vla4, vla2, groot, va, edge, nano and dreamzero. Cosmos Edge
and Nano require Python 3.13 and may share their vendor environment; the others
use Python 3.12. These recipes target Linux aarch64 / Jetson Thor. Install a
matching CUDA toolkit/native serving wheel for selected GPU kernels; a CPU
dependency check does not execute them. See INSTALL.rst for native builds and
REPRODUCE.rst for primary checkpoint preparation, actions and latency reports.
Install ``uv`` first, or pass its executable with ``--uv``. If the required
interpreter is missing, use ``uv python install 3.12`` (``3.13`` for Cosmos)
and pass the result of ``uv python find`` as ``--python``.
Replace the bootstrap's same-version pure ``flash_rt`` wheel explicitly with
``python -m pip install --force-reinstall --no-deps /path/to/native/flash_rt.whl``
and rerun ``python -m pip check``; an ordinary install can skip that replacement.

The bootstrap creates new source/environment directories, checks out a pinned
public Git commit, verifies the recorded source patch, and builds noneditable
vendor/core/adapter wheels. It never changes an existing installation. Source
licenses and notices accompany the patches. Each command records exact inputs,
package checks and failures. Both pip check and uv pip check must pass; no
--no-deps bypass is used. No weights or GPU operations run during bootstrap or
doctor. ``--defer-doctor`` records an explicit deferred CPU check.

All eight profiles now have fresh public package-install evidence; Edge/Nano
share the same qualified package stack. Six separate source checkouts passed
both package checks. The model-specific count and receipt hashes are recorded
in qualification.json. These package results are separate from native action,
WebSocket, latency and task-quality qualification. Historical pins.json records
observed interpreter precedence; bootstrap.json and inference_requirements.txt
are the actual public install recipe.

The shared public Torch 2.11.0 wheel imports as 2.11.0+cu130. Its public NVIDIA
cuSPARSELt 0.8.0 wheel has a packaging defect: its filename says aarch64 while
its internal WHEEL tag says sbsa. repair_vendor_wheel.py verifies the original
public artifact, changes only that tag and its RECORD entry, and checks every
native payload byte remains identical. wheel_repairs.json binds original and
repaired hashes. Both dependency checkers pass after this disclosed metadata
repair. Cosmos uses the exact public Torch 2.10.0+cu130, torchvision, torchcodec
and NATTEN wheels listed in its recipe; Python 3.12 extensions cannot be reused
in its Python 3.13 environment.

Inference-only packaging changes are explicit. Vendor metadata excludes
training/UI/robot-device and unrelated server/build requirements. VLA4/VLA2
preserve every LeRobot 0.4.4 implementation and license byte while declaring
only the pinned inference dependencies; 94 historical imported source files
match the public wheel. VLA4 uses Hugging Face Hub 0.35.3 instead of historical
0.33.5; both VLA recipes use cachebox 5.2.3 instead of inconsistent 6.2.5. The
VLA source patches retain the recorded attention/MoE implementations, so the
historical vendor baseline is not untouched upstream. These changes and their
scope are recorded in each recipe; they do not establish action equivalence.
VLA4's native package initializers also require pinned ipdb, torchdata and
pydantic. These were added after actual imports exposed the missing dependencies;
the resulting 127-package environment passes both checkers and its CPU FP8
doctor. Earlier failed import receipts remain preserved.
VLA2 likewise requires torchdata 0.11.0 and pydantic 2.13.4. Its complete
deployment module queries CUDA device properties while importing the native
grouped-GEMM tuning table. CPU doctor checks the mandatory dependencies and
exact source hashes, and reports that full import as deferred. A separate
actual-device import passed in the fresh environment; CPU doctor does not
claim that GPU gate or construct a model.
Cosmos's default native utility imports also require pinned boto3 and pandas.
Both are included in the shared Edge/Nano recipe. CPU doctor imports the actual
RoboLab policy-service entrypoint and checks its service/argument classes under
the offline GPU guard; it does not construct the model or load weights.
Only two explicitly audited pure Python sdists are allowed; the bootstrap does
not compile CUDA dependencies.

Cosmos native download tool
---------------------------

The unchanged Cosmos loader invokes ``uvx --with click hf@1.16.4 download``
even when its external VAE is already cached. Edge/Nano bootstrap therefore
installs uv 0.12.5 and prepares a separate HF CLI cache with 23 exact package
pins. Both the offline CLI reuse and that tool environment's package check
must pass. The model environment's older Hugging Face library is preserved. The complete
Triton 3.6.0 CPython 3.13 AArch64 wheel is separately hash-pinned to the
official PyTorch distribution: PyPI ships a different compiler binary under
the same version. Corrected-build inference qualification is recorded separately.
``activate.sh`` includes the tool environment variables and the correct uvx
binary directory; no login-shell PATH or historical user tool cache is needed.

For an existing compatible Python 3.13 environment, preparation is explicit::

    python scripts/prepare_native_tools.py prepare --root ./cosmos-hf-tool \
        --python "$VIRTUAL_ENV/bin/python" --uvx "$VIRTUAL_ENV/bin/uvx"
    source ./cosmos-hf-tool/run.env

This command downloads only public tool packages. It requires uvx 0.12.5;
the regular Edge/Nano bootstrap installs that version. A separate bounded
check can resolve an already prepared Wan VAE with both tool and Hub network
access disabled, without rehashing the large file::

    python scripts/prepare_native_tools.py probe --root ./cosmos-hf-tool \
        --cache-dir ./assets-edge/hf/hub --output ./cosmos-offline-check

``UV_OFFLINE=1`` prevents native subprocess package downloads at model load.
It does not set ``HF_HUB_OFFLINE`` in activation, so an explicit asset download
remains possible. The inference child or the explicit probe sets Hub offline
mode separately. Starting another bootstrap clears inherited UV settings
before its requested public package installation. Keep the prepared tool
directory and its Python environment together; deleting either invalidates
that tool preparation. No Cosmos runtime source, model computation, or
checkpoint bytes are changed.

Auxiliary assets
----------------

asset_profiles.json pins complete processor/tokenizer files and the external
Cosmos Wan2.2 VAE by repository revision, size and SHA256. DreamZero also
needs the original Wan2.1 T5/CLIP and Wan2.1 VAE initialization weights
(about 16.7 GB including tokenizer files), even though its primary checkpoint includes trained
component tensors: the unchanged native constructor loads those external
components first. Its pinned DROID configuration uses ``WanVideoVAE`` with
``z_dim=16`` in both native eager and Flash, so it selects Wan2.1 VAE.
``--include-primary`` validates these actual component targets against the
asset catalog before hashing the primary payload. These exact upstream LFS hashes are included in the same
asset preparation command; the original repositories' access and license
terms still apply. The helper verifies
source and destination bytes and writes a new isolated Hugging Face cache with
its own revision refs. QWEN25_PATH and QWEN3VL_PATH point to the exact prepared
processors. Original caches and their refs remain unchanged. Auto mode uses
hardlinks on the same filesystem and copies otherwise; linked payloads must
not be edited. PaliGemma requires original upstream repository access.
DreamZero activation sets ``NO_ALBUMENTATIONS_UPDATE=1`` so its preprocessing
package does not initiate a package-update HTTP request during inference.
Its native service entrypoint requires tyro 1.0.16, included with the historical
dependency constraints. CPU doctor imports the complete
``eval_utils.serve_dreamzero_wan22`` module and checks its policy/helper APIs
under the offline GPU guard; it does not construct a policy or load weights.

Activation permits the next explicit primary checkpoint download. Inference
reproduction children independently run offline. To use an existing cache and
also prepare the exact primary checkpoint in the new cache without duplicate
payload storage on the same filesystem::

    python scripts/prepare_auxiliary_assets.py prepare vla4 --root ./assets-vla4 \
        --cache-dir "$HOME/.cache/huggingface/hub" --local-files-only --include-primary
    source ./assets-vla4/run.env

``--include-primary`` binds the requested Hub commit reference, physical source
paths and freshly measured content hashes. Its primary file inventory is not
an independent upstream per-file digest certificate. Primary-only preparation
is also available through ``python -m benchmarks.regression.reproduce prepare``.
Keep VA's native tokenizer/text encoder/VAE subdirectories and Cosmos primary
processor metadata intact; no cross-model tokenizer or missing-asset
substitution is performed.
