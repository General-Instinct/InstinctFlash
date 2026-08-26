"""
FlashRT — High-performance VLA inference engine.

Public exports (stable API — see ``docs/stable_api.md``):

    flash_rt.load_model(...)   → VLAModel
    flash_rt.VLAModel          — unified inference wrapper

Supported models: Pi0.5, Pi0, Pi0-FAST, GROOT N1.6, GR00T N1.7, LingBot-VLA-V2.
Supported hardware: Jetson Thor (SM110), RTX 5090 (SM120), RTX 4090 (SM89); GR00T N1.7 and
LingBot-VLA-V2 add A100 (SM80) / H100 (SM90) paths.

The LingBot-VLA-V2 arm in this tree is the ``cuda_sm80`` / ``cuda_sm90`` upstream-BF16
datacenter graft: upstream kernels plus static-KV CUDA-graph replay and optional Triton
MoE/RMSNorm kernels (never on Thor). There is no SM110 dispatch row for it here; an engine
tier for launch-bound edge devices is available under commercial access.

Extending with new models: see ``docs/plugin_model_template.md``.

Usage::

    import flash_rt

    model = flash_rt.load_model(
        checkpoint="/path/to/checkpoint",
        framework="torch",
        autotune=3,
    )

    actions = model.predict(images=[base_img, wrist_img],
                            prompt="pick up the red block")
"""

__version__ = "0.1.0"

# ── Windows: register CUDA / cuDNN DLL search paths ──
# Python 3.8+ on Windows ignores PATH for C-extension dependencies
# (security hardening). The compiled .pyd needs cudart64_*.dll,
# cublas64_*.dll, cublasLt, cudnn — we add their canonical install
# directories to the secure DLL loader so `import flash_rt` works
# without the user pre-loading them. Linux is unaffected: this whole
# block is skipped via the sys.platform guard.
import os as _os
import sys as _sys
if _sys.platform == 'win32':
    _cuda_roots = [
        _os.environ.get('CUDA_PATH'),
        _os.environ.get('CUDA_PATH_V13_0'),
        _os.environ.get('CUDA_PATH_V12_9'),
        _os.environ.get('CUDA_PATH_V12_8'),
        _os.environ.get('CUDA_PATH_V12_4'),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
        _os.environ.get('CUDNN_PATH'),
    ]
    _seen = set()
    for _root in filter(None, _cuda_roots):
        for _sub in ('bin', r'extras\CUPTI\lib64', ''):
            _p = _os.path.join(_root, _sub) if _sub else _root
            if _p in _seen:
                continue
            _seen.add(_p)
            if _os.path.isdir(_p):
                try:
                    _os.add_dll_directory(_p)
                except (OSError, ValueError):
                    pass
    del _root, _sub, _p, _seen, _cuda_roots
del _os, _sys

from flash_rt.api import load_model, VLAModel  # noqa: E402

__all__ = ["load_model", "VLAModel"]
