# Vendor: openpi PyTorch pi0/pi0.5 model

## Source

Files in this directory are a snapshot of the upstream openpi PyTorch
modeling code:

| Local file | Source |
|---|---|
| `pi0_pytorch.py` | `openpi_src/src/openpi/models_pytorch/pi0_pytorch.py` |
| `gemma_pytorch.py` | `openpi_src/src/openpi/models_pytorch/gemma_pytorch.py` |
| `preprocessing_pytorch.py` | `openpi_src/src/openpi/models_pytorch/preprocessing_pytorch.py` |
| `image_tools.py` | `openpi_src/src/openpi/shared/image_tools.py` (PyTorch subset, JAX paths stripped) |
| `gemma_config.py` | shim replacing `openpi.models.gemma.get_config` (JAX-free) |
| `pi0_config.py` | shim replacing `openpi.models.pi0_config.Pi0Config` (JAX-free) |

Source repository: openpi (PyTorch port subtree).
Capture date: 2026-04-25.

## Why vendor

The upstream `openpi` package transitively depends on JAX, Flax, and
nnx — pulling those into the training container would inflate the
runtime footprint and risk version drift against the inference
framework. Only the PyTorch subset is needed by `training.trainers`,
so we snapshot it locally and rewrite the cross-package imports as
relative.

## Modifications from upstream

1. `from openpi.models.gemma import …` → `from . import gemma_config as _gemma`
2. `from openpi.models_pytorch.gemma_pytorch import …` → `from .gemma_pytorch import …`
3. `from openpi.shared import image_tools` → `from . import image_tools`
4. `image_tools.py` JAX functions removed; only `resize_with_pad_torch` retained.
5. `gemma_config.py`, `pi0_config.py`: thin JAX-free dataclass shims.

No semantic changes — model architecture, forward path, RMSNorm/RoPE
math, attention masking, and flow-matching loss are byte-identical to
upstream. The HF `transformers_replace/` patch is **not** vendored:
the container's installed `transformers==4.53.2` already has the patch
applied (verified via `transformers.models.siglip.check`).

## Extending the vendor

If `training/` adds dependencies on additional openpi modules,
snapshot them here and document the source in this table. Do **not**
import directly from an external openpi checkout — keep the vendor
tree as the single source of truth so the training stack remains
self-contained.
