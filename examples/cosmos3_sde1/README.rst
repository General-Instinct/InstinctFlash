Experimental Cosmos SDE1 reproduction
====================================

This separate package reproduces the historical Edge 270.24–271.35 ms and
Nano 714.96 ms operating points. They are SCREEN measurements: SDE1, CFG1,
native BF16 with NUMERIC cuDNN attention, zero padding and complete 32-by-8
action chunks. They do not inherit the original UniPC4 quality result and
have no task-quality certificate. The main Cosmos README table uses full
UniPC4/CFG3; task-quality evidence is reported separately.

Install this package into the separately bootstrapped Cosmos environment
after installing InstinctFlash and cosmos3-iwm. It contains eight exact
InstinctCompress helper files under their existing package names, licensed
AGPL-3.0-or-later; do not co-install a different package owning those same
``instinct_compress`` files. Model materials use their separate OpenMDW1.1
terms and retain the NVIDIA model card and notices in the overlay release.

Install from the source checkout into that Cosmos environment::

    python -m pip install ./examples/cosmos3_sde1

Inspect without importing Torch, downloading or creating files::

    python -m cosmos3_sde1 plan edge-seed12031
    python -m cosmos3_sde1 plan edge-seed12032
    python -m cosmos3_sde1 plan nano-original

Edge uses the original base revision
``3ea407af3e156c0af3b4bb6edd85842cc9a58777`` and one of two hash-bound
419,568,688-byte safe tensor overlays. These contain 28 absolute merged BF16
replacements each. No optimizer snapshot, pickle file, re-quantization or
merge rounding is needed. The download manifest records the exact SHA256.
The recipes specify the hash-bound overlay files in the ``thor-2026-09-15``
release. Fetch, prepare and run with::

    python -m cosmos3_sde1 fetch edge-seed12031 --output downloads-edge31
    python -m cosmos3_sde1 prepare edge-seed12031 --downloads downloads-edge31 --output prepared-edge31
    python -m cosmos3_sde1 run edge-seed12031 --checkpoint prepared-edge31 --output result-edge31

``fetch`` records local base/overlay paths in ``downloads.json`` and fetches
the pinned external Wan2.2 VAE. ``prepare --downloads`` validates that bound
download receipt and resolves its paths automatically, then creates an
independent checkpoint, and verifies the final modified shard against the
original exported whole-shard hash. Repeat with ``edge-seed12032`` and its
corresponding overlay. Original checkpoints and exports remain untouched.

Nano uses unchanged original weights at revision
``2b9f9517efcfbf26e222945b386ae9b65c0930ac`` and needs no trained overlay::

    python -m cosmos3_sde1 fetch nano-original --output downloads-nano
    python -m cosmos3_sde1 prepare nano-original --downloads downloads-nano --output prepared-nano
    python -m cosmos3_sde1 run nano-original --checkpoint prepared-nano --output result-nano

The Nano recipe changes only the declared sampler/configuration and enables
the historical persistent text cache plus shared SwiGLU implementation. Its
12 warmup requests prepare two prompt slots; all 24 timed requests must replay
36 layers without graph preparation. New-prompt preparation is outside this
steady-state latency. Edge uses six warmup and ten measured requests.

Already-downloaded local files can instead be supplied through ``prepare --base``
and, for Edge, ``--overlay``. These manual inputs are mutually exclusive with
``--downloads`` and undergo the same checkpoint/tensor hash checks.

Preparation copies files by default. To save disk space, add ``--link-unmodified``
when the output and original weight files are on the same filesystem. This
hardlinks only unchanged safetensors weights; configuration, sidecars and Edge's
modified shard are separate copies. The command verifies the source and linked
file hashes and fails across filesystems without a copy fallback. Treat linked
weights as immutable: editing either path edits the shared inode. Use the default
copy mode if you need independent editable weights. For example::

    python -m cosmos3_sde1 prepare nano-original --downloads downloads-nano \
      --link-unmodified --output prepared-nano

Each run uses a fresh installed-package child process, the exact decoded
non-pickle public fixture, fixed historical request seeds and action checks.
Downloads are an explicit phase; the benchmark child runs offline. Outputs
are created once, errors and logs are retained, and there is no silent retry.
The benchmark requires a Thor SM110 GPU and the recorded compatible Cosmos
vendor environment, Torch 2.10/CUDA13 and cuDNN9.15.1. A new installed run
records its actual source/runtime identity; it is not relabeled as the old
September12 binary/source snapshot. CPU assembly tests do not qualify speed,
GPU inference, serving or downstream task success.
