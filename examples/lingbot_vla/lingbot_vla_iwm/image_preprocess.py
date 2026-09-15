"""Bitexact GPU-side Qwen image normalization for LingBot-VLA-4B.

The official server's PIL bilinear camera resize and Qwen slow-processor patch layout remain on
CPU because replacing either interpolation path changes pixels. The slow processor stops before
rescale/normalize, writes the three fixed patch tensors into reusable pinned storage, and one H2D
copy feeds the unchanged per-channel Qwen formula on CUDA. The first six live batches compare
the resulting BF16 image tensor with upstream preprocessing exactly; a mismatch restores the
original function immediately.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


def _unwrap_inactive_prepare(prepare):
    while (owner := getattr(prepare, "_ifl_preprocess_owner", None)) is not None:
        if owner.server is not None and not owner.rejected:
            break
        prepare = owner.original_prepare_images
    return prepare


def _upstream_prepare(prepare):
    while (owner := getattr(prepare, "_ifl_preprocess_owner", None)) is not None:
        prepare = owner.original_prepare_images
    return prepare


@dataclass
class GPUImagePreprocess:
    server: object
    original_prepare_images: object
    installed_prepare_images: object
    device: str
    checks: int = 0
    max_abs_delta: float = 0.0
    passed: bool = False
    rejected: bool = False
    calls: int = 0

    def close(self) -> None:
        from lingbotvla.data.vla_data import utils as utils_module

        self.rejected = True
        if utils_module.prepare_images is self.installed_prepare_images:
            utils_module.prepare_images = _unwrap_inactive_prepare(self.original_prepare_images)
        release = getattr(self.installed_prepare_images, "_ifl_release_storage", None)
        if release is not None:
            release()
        server = self.server
        if (
            server is not None
            and getattr(server, "_instinctflash_gpu_preprocess", None) is self
        ):
            delattr(server, "_instinctflash_gpu_preprocess")
        self.server = None


def install_gpu_image_preprocess(
    server,
    *,
    device: str = "cuda",
) -> GPUImagePreprocess:
    current = getattr(server, "_instinctflash_gpu_preprocess", None)
    if current is not None:
        return current

    import sys

    from lingbotvla.data.vla_data import utils as utils_module
    from lingbotvla.data.vla_data.transform import resize_with_pad_item

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError(
            f"LingBot-VLA-4B GPU preprocessing requires CUDA, got {device!r}"
        )

    processor = server.processor.image_processor
    if not (bool(processor.do_rescale) and bool(processor.do_normalize)):
        raise ValueError("Qwen image processor must enable rescale and normalize")
    image_mean = tuple(float(value) for value in processor.image_mean)
    image_std = tuple(float(value) for value in processor.image_std)
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("Qwen image processor must have three-channel mean/std")

    original_prepare = _unwrap_inactive_prepare(utils_module.prepare_images)
    reference_prepare = _upstream_prepare(original_prepare)
    lock = threading.Lock()
    staging = None
    device_raw = None
    mean_vector = None
    std_vector = None

    report = GPUImagePreprocess(
        server=server,
        original_prepare_images=original_prepare,
        installed_prepare_images=None,
        device=str(target_device),
    )

    def release_storage():
        nonlocal staging, device_raw, mean_vector, std_vector
        with lock:
            staging = device_raw = mean_vector = std_vector = None

    def reject(reason: str) -> None:
        report.rejected = True
        report.passed = False
        if utils_module.prepare_images is gpu_prepare_images:
            utils_module.prepare_images = _unwrap_inactive_prepare(original_prepare)
        release_storage()
        print(
            f"[LingBot-VLA-4B GPU preprocess] rejected ({reason}); "
            "upstream CPU preprocessing continues.",
            file=sys.stderr,
            flush=True,
        )

    def candidate(observation, resize_imgs_with_padding, keys):
        nonlocal staging, device_raw, mean_vector, std_vector

        raw_values = []
        masks = []
        for key in keys:
            if key not in observation["image"]:
                raise RuntimeError(f"missing image feature {key!r}")
            image = resize_with_pad_item(
                observation["image"][key],
                *resize_imgs_with_padding,
                pad_value=0,
            )
            raw = processor(
                image,
                do_rescale=False,
                do_normalize=False,
            )["pixel_values"]
            raw_values.append(torch.as_tensor(raw))
            masks.append(True)

        signature = (
            len(raw_values),
            tuple(raw_values[0].shape),
            raw_values[0].dtype,
        )
        if any(tuple(value.shape) != signature[1] for value in raw_values) or any(
            value.dtype != signature[2] for value in raw_values
        ):
            raise RuntimeError("Qwen patch tensors do not share one fixed signature")

        with lock:
            if staging is None:
                staging = torch.empty(
                    (signature[0], *signature[1]),
                    dtype=signature[2],
                    pin_memory=True,
                )
                device_raw = torch.empty_like(staging, device=target_device)
                feature_shape = (
                    3,
                    int(processor.temporal_patch_size),
                    int(processor.patch_size),
                    int(processor.patch_size),
                )
                if (
                    feature_shape[0]
                    * feature_shape[1]
                    * feature_shape[2]
                    * feature_shape[3]
                    != signature[1][-1]
                ):
                    raise RuntimeError(
                        f"unexpected Qwen patch width {signature[1][-1]}"
                    )
                mean = torch.tensor(
                    image_mean,
                    device=target_device,
                    dtype=torch.float32,
                )
                std = torch.tensor(
                    image_std,
                    device=target_device,
                    dtype=torch.float32,
                )
                mean_vector = mean.view(3, 1, 1, 1).expand(feature_shape).reshape(-1)
                std_vector = std.view(3, 1, 1, 1).expand(feature_shape).reshape(-1)
            elif (
                tuple(staging.shape) != (signature[0], *signature[1])
                or staging.dtype != signature[2]
            ):
                raise RuntimeError("Qwen patch signature changed after warmup")

            for index, value in enumerate(raw_values):
                staging[index].copy_(value)
            device_raw.copy_(staging, non_blocking=True)
            images = (
                (device_raw * float(processor.rescale_factor) - mean_vector)
                / std_vector
            ).to(torch.bfloat16)

        return images, torch.tensor(masks, dtype=torch.bool), []

    def gpu_prepare_images(
        image_processor,
        observation,
        resize_imgs_with_padding,
        use_depth_align=False,
        image_keys=None,
    ):
        if report.rejected:
            return original_prepare(
                image_processor,
                observation,
                resize_imgs_with_padding,
                use_depth_align=use_depth_align,
                image_keys=image_keys,
            )
        keys = tuple(image_keys or ())
        if image_processor is not processor or use_depth_align or len(keys) != 3:
            return original_prepare(
                image_processor,
                observation,
                resize_imgs_with_padding,
                use_depth_align=use_depth_align,
                image_keys=image_keys,
            )

        try:
            result = candidate(
                observation,
                resize_imgs_with_padding,
                keys,
            )
        except Exception as error:  # noqa: BLE001 - upstream fallback is the contract
            reject(f"{type(error).__name__}: {error}")
            return original_prepare(
                image_processor,
                observation,
                resize_imgs_with_padding,
                use_depth_align=use_depth_align,
                image_keys=image_keys,
            )

        if report.checks < 6:
            reference = reference_prepare(
                image_processor,
                observation,
                resize_imgs_with_padding,
                use_depth_align=False,
                image_keys=keys,
            )
            reference_images = reference[0].to(
                device=target_device,
                dtype=torch.bfloat16,
            )
            from instinctflash.runtime.capture_self_check import compare_tensors
            image_check = compare_tensors(reference_images, result[0])
            mask_check = compare_tensors(reference[1], result[1])
            delta = image_check["max_abs_delta"]
            masks_equal = mask_check["valid"] and mask_check["bitexact"]
            report.max_abs_delta = max(report.max_abs_delta, delta)
            report.checks += 1
            if not (image_check["valid"] and image_check["bitexact"] and masks_equal):
                reject(
                    f"live self-check delta {delta:.3e}, " f"masks_equal={masks_equal}"
                )
                return reference
            if report.checks == 6:
                report.passed = True
                print(
                    "[LingBot-VLA-4B GPU preprocess] bit-exact on 6 live batches; "
                    "pinned H2D + CUDA normalize enabled."
                )

        report.calls += 1
        return result

    gpu_prepare_images._ifl_preprocess_owner = report
    gpu_prepare_images._ifl_release_storage = release_storage
    report.installed_prepare_images = gpu_prepare_images
    utils_module.prepare_images = gpu_prepare_images
    server._instinctflash_gpu_preprocess = report
    return report


__all__ = ["GPUImagePreprocess", "install_gpu_image_preprocess"]
