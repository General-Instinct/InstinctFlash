InstinctFlash full-source release
================================

The public repository includes the Runtime, CLI, model adapters, FlashRT
frontends, CUDA kernels, training and distillation code, and benchmark tools.
There is one public execution interface. No Instinct account or private
accelerator package is required.

Start with the `RTX 4090 <../INSTALL.rst#rtx-4090>`_ or
`Jetson Thor <../INSTALL.rst#jetson-thor>`_ installation profile and the
`reproduction guide <../REPRODUCE.rst>`_. They cover a fresh vendor environment,
pinned checkpoint and auxiliary downloads, paired inference measurements,
and actual CLI/WebSocket calls.
The benchmark selects eight model variants; LingBot-VA at 2V/4A is an additional
operating point. Compatible fine-tunes use their family's adapter and need their
own action and task evaluation.

Source, binaries and model assets
--------------------------------

The root package and adapters retain the root AGPL license. FlashRT retains its
Apache license; bundled third-party code and native wheels carry their own
notices. Upstream model checkpoints, tokenizers and experimental student
overlays retain their original access conditions and licenses. Checkpoint
weights are obtained separately from source installation.

The published native artifacts and `FlashRT source-build instructions
<../serving/README.rst>`_ target Jetson Thor. Its Python 3.12 native wheel is
specific to Linux aarch64 and the recorded Torch/CUDA stack. Edge and Nano use
Python 3.13 and Thor's separate BF16 C library. For RTX 4090, use the
``--target rtx4090`` bootstrap: its Linux x86-64 profiles select the SM89
PyTorch/Triton execution path described in the installation guide.

Native precision with a BITEXACT transformation ceiling is the default.
NUMERIC, FP8 and sampling/cache changes are explicit choices through the same
Runtime. Installing a backend does not select any of those choices.

Reproducible evidence
---------------------

The public reproduction commands retain the exact checkpoint revisions, inputs,
source identities, selected options, actions and timing samples. Each measured
arm starts in a fresh process. The serving smoke invokes the installed CLI and
checks six requests across two episodes, including history, feedback and reset.

README latency rows describe particular hardware, checkpoints and execution
policies. They do not establish task accuracy, every fine-tune's behavior, or
performance on another device. Alternative-framework recipes retain their own
step and history protocols. Experimental SDE1 results remain separate from the
full UniPC4/CFG3 Cosmos selection.

Full RTX 4090 qualification remains pending. The
`RTX 4090 catalog <rtx4090/deployment_profiles.json>`_ records target status;
prepared profiles and CPU packaging checks do not establish completed GPU
measurements. The historical Thor results remain separate.

The recorded RGB fixture has its RoboTwin license and attribution alongside
it. Bulk model weights, machine-local caches and unreviewed simulator assets
are not embedded in the source distribution. Historical evidence is selected
individually with its inputs, limitations and failed attempts preserved.

Build a source and wheel candidate
---------------------------------

From the repository root, using the documented development dependencies::

    python scripts/build_public_release.py --scope full \
      --output /tmp/instinctflash-public-candidate --build-wheels

The output directory must be new. The builder checks selected source hashes,
materializes reviewed in-repository symlinks, builds nine Python distributions
and verifies their contents. It excludes downloaded build trees and model
weights; native CUDA binaries have their own source-build procedure.

A successful CPU build checks packaging, not inference speed or task success.
Run the reproduction and serving commands in each model's pinned environment
before making a device-specific claim. ``oss_scope.json`` preserves the earlier,
superseded core-only classification for audit history; ``--scope full`` is the
current complete source selection.
