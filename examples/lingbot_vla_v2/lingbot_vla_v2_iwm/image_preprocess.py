"""GPU image preprocessing prototype for LingBot-VLA-V2 inference.

The upstream server resizes the three RobotWin cameras independently on CPU, then invokes the
Qwen fast image processor three times on CPU. The validated default batches the three otherwise
identical CPU resizes without changing their values, transfers the fixed-size result through
reusable pinned storage, and lets the unchanged Qwen processor normalize and patchify on CUDA. An
experimental ``full`` mode also moves resize to CUDA, but is opt-in because its interpolation
changes policy numerics slightly.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
import torch
from torchvision.transforms.v2 import Resize


@dataclass
class GPUImagePreprocessInstall:
    server: object
    original_resize_image: object
    camera_count: int
    input_hw: tuple[int, int]
    output_hw: tuple[int, int]
    device: str
    mode: str

    def close(self) -> None:
        server = self.server
        if server is not None and getattr(server, "_instinctflash_gpu_resize", None) is self:
            server.resize_image = self.original_resize_image
            delattr(server, "_instinctflash_gpu_resize")
        self.server = None


def install_lingbot_gpu_image_preprocess(
    server,
    *,
    device: str = "cuda",
    mode: str = "processor",
) -> GPUImagePreprocessInstall:
    """Move fixed-shape image processing to CUDA while retaining Qwen's processor.

    Unsupported image layouts and dtypes fall back to the original preprocessing method. The
    published RobotWin contract is three uint8 HWC images with identical 480x640 resolution.
    """
    if mode not in {"full", "processor"}:
        raise ValueError("LingBot GPU preprocessing mode must be 'full' or 'processor'")
    current = getattr(server, "_instinctflash_gpu_resize", None)
    if current is not None:
        if current.mode != mode:
            raise RuntimeError(
                f"GPU preprocessing is already installed in {current.mode!r} mode"
            )
        return current

    image_features = tuple(server.vla.feature_transform.org_features["images"])
    if len(image_features) != 3:
        raise ValueError(
            f"LingBot GPU preprocessing expects three camera features, got {len(image_features)}"
        )
    image_size = int(getattr(server.data_config, "img_size", 256))
    output_hw = (image_size, image_size)
    input_hw = (480, 640)
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError(f"LingBot GPU preprocessing requires CUDA, got {device!r}")

    original = server.resize_image
    resize = Resize(output_hw)
    staging_u8 = device_u8 = None
    staging_f32 = device_f32 = None
    if mode == "full":
        staging_u8 = torch.empty(
            (len(image_features), 3, *input_hw), dtype=torch.uint8, pin_memory=True
        )
        device_u8 = torch.empty_like(staging_u8, device=target_device)
    else:
        staging_f32 = torch.empty(
            (len(image_features), 3, *output_hw), dtype=torch.float32, pin_memory=True
        )
        device_f32 = torch.empty_like(staging_f32, device=target_device)
    lock = threading.Lock()

    report = GPUImagePreprocessInstall(
        server=server,
        original_resize_image=original,
        camera_count=len(image_features),
        input_hw=input_hw,
        output_hw=output_hw,
        device=str(target_device),
        mode=mode,
    )

    def fast_resize_image(observation):
        images = [observation.get(key) for key in image_features]
        if mode == "processor":
            # Resize all cameras as one NCHW batch. Torch's CPU kernel is elementwise-identical to
            # the upstream three independent calls, while exposing the camera dimension to its
            # parallel scheduler. Unsupported inputs retain the exact upstream fallback.
            supported = all(
                isinstance(image, np.ndarray)
                and image.dtype == np.uint8
                and image.shape == (*input_hw, 3)
                for image in images
            )
            if supported:
                cpu_batch = torch.stack(
                    [torch.as_tensor(image).permute(2, 0, 1).contiguous() for image in images]
                ).to(dtype=torch.float32)
                resized_batch = resize(cpu_batch)
                resized_images = list(resized_batch.unbind(0))
            else:
                original(observation)
                resized_images = [observation.get(key) for key in image_features]
            supported_resized = all(
                isinstance(image, torch.Tensor)
                and image.device.type == "cpu"
                and image.dtype == torch.float32
                and image.shape == (3, *output_hw)
                for image in resized_images
            )
            if not supported_resized:
                return
            with lock:
                for index, image in enumerate(resized_images):
                    staging_f32[index].copy_(image)
                device_f32.copy_(staging_f32, non_blocking=True)
                for index, key in enumerate(image_features):
                    observation[key] = device_f32[index]
            return

        supported = all(
            isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and image.shape == (*input_hw, 3)
            for image in images
        )
        if not supported:
            return original(observation)

        # The loop only copies 2.8 MiB into an allocation made once at load time. H2D then runs
        # asynchronously; model inference synchronizes naturally when actions are copied to CPU.
        with lock:
            for index, image in enumerate(images):
                staging_u8[index].copy_(torch.from_numpy(image).permute(2, 0, 1))
            device_u8.copy_(staging_u8, non_blocking=True)
            resized = resize(device_u8.to(dtype=torch.float32))
            for index, key in enumerate(image_features):
                observation[key] = resized[index]

    server.resize_image = fast_resize_image
    server._instinctflash_gpu_resize = report
    return report


__all__ = ["GPUImagePreprocessInstall", "install_lingbot_gpu_image_preprocess"]
