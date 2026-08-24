"""Frame → pi0.5 Observation adapter (LIBERO is the default mapping).

The pi0.5 ``embed_prefix`` expects **3** image entries keyed by
the openpi camera naming (``base_0_rgb``, ``left_wrist_0_rgb``,
``right_wrist_0_rgb``); concrete datasets typically ship a
different set + naming and need a remap. Two halves:

* :func:`decoded_to_observation` — dataset-agnostic. Takes a
  decoded ``{pi05_camera_key: uint8 (B,H,W,3) | None}`` dict and
  returns the :class:`Pi05Observation`. ``None`` slots get zero
  fill + a False mask. The generic RECAP driver calls this.

* :func:`decode_frame_images` / :func:`frames_to_observation` —
  the host-side decode loop, parametrised by a
  ``camera_map: dict[pi05_key -> source_key | None]``. Default is
  :data:`LIBERO_CAMERA_MAP` (LIBERO's 2-cam layout); pass your
  own map to plug in another dataset without touching this
  module.

Token masks (``token_ar_mask`` / ``token_loss_mask``, both zero)
match the openpi LIBERO finetune convention.

State convention: pi0.5's ``embed_suffix`` slot expects state of
``action_dim=32``. LIBERO state is 8-dim, so we left-pad with
zeros up to 32. Same approach as the openpi-compiler train script.
"""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Sequence

import numpy as np
import torch

from training.rl.lerobot_libero import LeRobotFrame

PI05_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
PI05_STATE_DIM = 32

# pi0.5 camera key → source camera key in the dataset's
# ``frame.image_bytes``. ``None`` means the pi0.5 slot is filled
# with zeros and ``image_masks[<key>] = False``.
#
# LIBERO ships 2 cameras (``observation.image`` and
# ``observation.wrist_image``); pi0.5 expects 3 → fill the right
# wrist with zeros + a False mask, matching upstream
# train_jax_lora_recap.py. Other datasets pass their own
# ``camera_map`` to :func:`decode_frame_images` /
# :func:`frames_to_observation`.
LIBERO_CAMERA_MAP: dict[str, str | None] = {
    "base_0_rgb":         "observation.image",
    "left_wrist_0_rgb":   "observation.wrist_image",
    "right_wrist_0_rgb":  None,
}

# Back-compat alias — old name kept for any external caller. New
# code should reference ``LIBERO_CAMERA_MAP`` (public) and pass a
# different map for non-LIBERO datasets.
_LIBERO_TO_PI05 = LIBERO_CAMERA_MAP


@dataclasses.dataclass
class Pi05Observation:
    """Plain-Python Observation matching the vendored Pi05 expectations.

    Fields mirror what
    ``training/_vendor/openpi_pi0_pytorch/preprocessing_pytorch.py``
    accesses, all on the same CUDA device.
    """

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor
    token_loss_mask: torch.Tensor


def _decode_image(b: bytes) -> np.ndarray:
    """PNG/JPEG bytes → ``uint8 (H, W, 3)``."""
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    if arr.shape != (224, 224, 3):
        raise ValueError(f"unexpected image shape {arr.shape}; expected (224,224,3)")
    return arr


def _stack_camera(frames: Sequence[LeRobotFrame], source_key: str) -> np.ndarray:
    """Stack one camera across the batch into ``uint8 (B, H, W, 3)``."""
    return np.stack(
        [_decode_image(f.image_bytes[source_key]) for f in frames],
        axis=0,
    )


def decode_frame_images(
    frames: Sequence[LeRobotFrame],
    camera_map: dict[str, str | None] | None = None,
) -> dict[str, np.ndarray | None]:
    """Decode each source camera into a stacked ``uint8 (B, H, W, 3)`` array.

    Returns one entry per pi0.5 camera key. ``None`` means the slot
    has no source camera (e.g. LIBERO has no right wrist) and
    downstream should fill zeros + a False mask. Pure-CPU; callable
    from a DataLoader worker.

    Args:
        frames: Per-batch sequence of frames carrying ``image_bytes``.
        camera_map: ``{pi05_key: source_key | None}``. Defaults to
            :data:`LIBERO_CAMERA_MAP`. Pass a different map to plug
            a non-LIBERO dataset into the generic RECAP loader.
    """
    cmap = camera_map if camera_map is not None else LIBERO_CAMERA_MAP
    out: dict[str, np.ndarray | None] = {}
    for pi_key, source_key in cmap.items():
        out[pi_key] = None if source_key is None else _stack_camera(frames, source_key)
    return out


def pad_states(states_np: np.ndarray) -> np.ndarray:
    """Right-pad a ``(B, state_dim)`` state batch to pi0.5's 32-dim."""
    if states_np.ndim != 2:
        raise ValueError(f"states must be 2D, got shape {states_np.shape}")
    if states_np.shape[1] < PI05_STATE_DIM:
        pad = np.zeros(
            (states_np.shape[0], PI05_STATE_DIM - states_np.shape[1]),
            dtype=np.float32,
        )
        return np.concatenate([states_np.astype(np.float32, copy=False), pad], axis=1)
    if states_np.shape[1] > PI05_STATE_DIM:
        raise ValueError(
            f"state dim {states_np.shape[1]} > pi0.5 state_dim {PI05_STATE_DIM}"
        )
    return states_np.astype(np.float32, copy=False)


def decoded_to_observation(
    decoded_images: dict[str, np.ndarray | None],
    states_padded_np: np.ndarray,
    *,
    tokenized_prompt: torch.Tensor,
    tokenized_prompt_mask: torch.Tensor,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Pi05Observation:
    """Assemble a :class:`Pi05Observation` from already-decoded NHWC uint8 images.

    Splits the host-side image decode + state pad (pure-CPU) from
    the H2D + dtype conversion (CUDA), so the former can run inside
    a DataLoader worker while the latter stays on the main thread.

    Args:
        decoded_images: ``{pi05_camera_key: uint8 (B, H, W, 3) | None}``;
            ``None`` slots are filled with zeros + ``False`` mask.
        states_padded_np: ``float32 (B, 32)`` already padded to
            ``PI05_STATE_DIM`` (use :func:`pad_states`).
        tokenized_prompt, tokenized_prompt_mask: as in
            :func:`frames_to_observation`.
        device, dtype: target device / float dtype.
    """
    if states_padded_np.shape[1] != PI05_STATE_DIM:
        raise ValueError(
            f"states_padded_np must have last dim {PI05_STATE_DIM}, "
            f"got {states_padded_np.shape}"
        )
    bsize = states_padded_np.shape[0]
    device = torch.device(device)

    images: dict[str, torch.Tensor] = {}
    image_masks: dict[str, torch.Tensor] = {}
    for pi_key in PI05_CAMERA_KEYS:
        arr = decoded_images.get(pi_key)
        if arr is None:
            zero = torch.empty(bsize, 3, 224, 224, dtype=dtype, device=device)
            zero.fill_(-1.0)
            images[pi_key] = zero
            image_masks[pi_key] = torch.zeros(bsize, dtype=torch.bool, device=device)
            continue
        if arr.shape != (bsize, 224, 224, 3):
            raise ValueError(
                f"decoded image[{pi_key!r}] shape {arr.shape} != "
                f"({bsize}, 224, 224, 3)"
            )
        f = arr.astype(np.float32) / 255.0
        f = f * 2.0 - 1.0
        t = torch.from_numpy(f).permute(0, 3, 1, 2).contiguous().to(dtype=dtype, device=device)
        images[pi_key] = t
        image_masks[pi_key] = torch.ones(bsize, dtype=torch.bool, device=device)

    state = torch.from_numpy(states_padded_np).to(dtype=dtype, device=device)

    if tokenized_prompt.shape[0] != bsize:
        raise ValueError(
            f"tokenized_prompt batch {tokenized_prompt.shape[0]} != batch {bsize}"
        )
    if tokenized_prompt.shape != tokenized_prompt_mask.shape:
        raise ValueError(
            f"tokenized_prompt {tuple(tokenized_prompt.shape)} != "
            f"tokenized_prompt_mask {tuple(tokenized_prompt_mask.shape)}"
        )

    token_ar_mask = torch.zeros_like(tokenized_prompt, dtype=torch.long)
    token_loss_mask = torch.zeros_like(tokenized_prompt, dtype=torch.bool)

    return Pi05Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt.to(device),
        tokenized_prompt_mask=tokenized_prompt_mask.to(device),
        token_ar_mask=token_ar_mask.to(device),
        token_loss_mask=token_loss_mask.to(device),
    )


def frames_to_observation(
    frames: Sequence[LeRobotFrame],
    *,
    tokenized_prompt: torch.Tensor,
    tokenized_prompt_mask: torch.Tensor,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    camera_map: dict[str, str | None] | None = None,
) -> Pi05Observation:
    """Build a pi0.5 ``Observation`` from a batch of frames.

    Convenience wrapper that does the host-side decode + state pad
    inline and then calls :func:`decoded_to_observation`. The async
    DataLoader path in ``training/rl/async_loader.py`` calls the two
    halves separately so the decode runs in a worker process.

    Args:
        frames: Sequence of frames (one per batch item) carrying
            ``image_bytes`` keyed by the dataset's camera names.
        tokenized_prompt: ``int64[B, L]`` — output of
            ``PaligemmaTokenizer(...)`` already.
        tokenized_prompt_mask: ``bool[B, L]``.
        device, dtype: Target device / float dtype for the image and
            state tensors. Pi05 backbone runs at ``float32`` in the
            test fixtures and ``bfloat16`` in production — caller picks.
        camera_map: ``{pi05_key: source_key | None}``. Defaults to
            :data:`LIBERO_CAMERA_MAP`. Pass a different map to plug a
            non-LIBERO dataset into the generic RECAP loader.
    """
    if len(frames) == 0:
        raise ValueError("frames must be non-empty")
    decoded = decode_frame_images(frames, camera_map=camera_map)
    states_np = np.stack([f.state for f in frames], axis=0).astype(np.float32)
    states_pad = pad_states(states_np)
    return decoded_to_observation(
        decoded,
        states_pad,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        device=device,
        dtype=dtype,
    )


def pad_actions_to_horizon(
    actions: np.ndarray, *, action_horizon: int, action_dim: int
) -> np.ndarray:
    """Right-pad / truncate a single action vector to ``(action_horizon, action_dim)``.

    LIBERO frames carry one action per frame. Pi0.5's flow-matching
    loss expects an action *chunk* of length ``action_horizon``.
    The simplest synthesis at this prototype scale is to repeat the
    one ground-truth action across the chunk and right-pad the
    action_dim with zeros (pi0.5 action_dim=32 vs LIBERO 7).
    Production data loaders chunk consecutive frames; this helper
    is a stop-gap for the W8 single-step plumbing test.
    """
    out = np.zeros((action_horizon, action_dim), dtype=np.float32)
    out[:, : actions.shape[0]] = actions[None, :]
    return out
