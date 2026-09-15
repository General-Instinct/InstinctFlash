"""Bitexact GPU-side collation for GR00T N1.7 on SM120.

Qwen's bicubic resize is intentionally left on CPU: torchvision's CPU and CUDA bicubic
implementations differ for real images and can change BF16 values. The exact CPU path instead
stops after uint8 resize/patchify, halves H2D image bytes versus BF16 staging, and performs
the uniform Qwen normalization plus BF16 conversion on GPU. Prompt/grid tensors are cached on
device and state tensors are stacked directly, removing NumPy round-trips and repeated device
allocations for static fields.

The first six live batches are compared field-for-field with upstream collation after its exact
BF16/device conversion. Any mismatch permanently restores upstream collation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

FAMILY = "GR00T N1.7 GPU collate"


def _tensor_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    from instinctflash.runtime.capture_self_check import compare_tensors
    verdict = compare_tensors(left, right)
    if verdict["valid"] and verdict["bitexact"]:
        return 0.0
    if not verdict["valid"]:
        return float("inf")
    return float(verdict["max_abs_delta"]) or float("inf")


class GPUCollator:
    """Run exact Qwen normalization/staging on GPU with a six-batch live gate."""

    SELF_CHECK_INPUTS = 6
    MAX_STATIC_SIGNATURES = 8

    def __init__(self, policy, *, device: torch.device) -> None:
        self.policy = policy
        self.original = policy.collate_fn
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.qwen = self.original.processor
        self.image_processor = self.qwen.image_processor
        self.token_cache: OrderedDict[tuple, dict[str, torch.Tensor]] = OrderedDict()
        self.static_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self.checks = 0
        self.max_abs_delta = 0.0
        self.rejected = False
        self.passed = False
        self.calls = 0

        means = tuple(float(value) for value in self.image_processor.image_mean)
        stds = tuple(float(value) for value in self.image_processor.image_std)
        factor = float(self.image_processor.rescale_factor)
        supported = (
            bool(self.image_processor.do_rescale)
            and bool(self.image_processor.do_normalize)
            and means == (0.5, 0.5, 0.5)
            and stds == (0.5, 0.5, 0.5)
            and factor == 1.0 / 255.0
        )
        if not supported:
            raise ValueError(
                "GPU collate requires Qwen mean/std=0.5 and rescale_factor=1/255"
            )
        # This is the exact fused mean/std constructed by BaseImageProcessorFast.
        self._byte_mean = means[0] * (1.0 / factor)
        self._byte_std = stds[0] * (1.0 / factor)

    @staticmethod
    def _batch_feature(values):
        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(data={"inputs": values})

    @staticmethod
    def _stack(values):
        if values and all(torch.is_tensor(value) for value in values):
            return torch.stack(values)
        return torch.from_numpy(np.stack(values))

    def _remember(self, cache: OrderedDict, key: tuple, value: Any):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.MAX_STATIC_SIGNATURES:
            cache.popitem(last=False)
        return value

    def _static_device(self, name: str, value: torch.Tensor) -> torch.Tensor:
        cpu = value.detach().contiguous().cpu()
        key = (
            name,
            tuple(cpu.shape),
            cpu.dtype,
            cpu.numpy().tobytes(),
        )
        cached = self.static_cache.get(key)
        if cached is not None:
            self.static_cache.move_to_end(key)
            return cached
        return self._remember(self.static_cache, key, cpu.to(self.device))

    def _image_and_tokens(self, contents):
        texts = [content["text"] for content in contents]
        images = [image for content in contents for image in content["images"]]

        # Preserve CPU bicubic exactly, but stop before the 11 MiB float32 normalize output.
        raw = self.image_processor(
            images=images,
            return_tensors="pt",
            do_rescale=False,
            do_normalize=False,
        )
        raw_pixels = raw["pixel_values"]
        image_grid = raw["image_grid_thw"]
        if raw_pixels.dtype != torch.uint8:
            raise RuntimeError(f"expected uint8 Qwen patches, got {raw_pixels.dtype}")

        token_key = (
            tuple(texts),
            tuple(image_grid.shape),
            tuple(image_grid.reshape(-1).tolist()),
        )
        tokens = self.token_cache.get(token_key)
        if tokens is None:
            # The cache-miss path delegates placeholder expansion/tokenization to Qwen.
            # It intentionally pays one duplicate uint8 resize per prompt/grid signature;
            # all steady-state batches use only the image processor above.
            full = self.qwen(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                images_kwargs={
                    "do_rescale": False,
                    "do_normalize": False,
                },
            )
            if not torch.equal(raw_pixels, full["pixel_values"]):
                raise RuntimeError(
                    "Qwen raw image path changed between equivalent calls"
                )
            if not torch.equal(image_grid, full["image_grid_thw"]):
                raise RuntimeError("Qwen image grid changed between equivalent calls")
            tokens = {
                "input_ids": full["input_ids"].to(self.device),
                "attention_mask": full["attention_mask"].to(self.device),
                "image_grid_thw": full["image_grid_thw"].to(self.device),
            }
            self._remember(self.token_cache, token_key, tokens)
        else:
            self.token_cache.move_to_end(token_key)

        # Normalization is elementwise and Qwen's three channels share mean/std, so it
        # commutes with patch permutation. BF16 results are exactly equal to upstream's
        # CPU normalize -> CPU BF16 -> H2D path, as enforced by the live gate below.
        pixel_values = raw_pixels.to(self.device, dtype=torch.float32)
        pixel_values.sub_(self._byte_mean).div_(self._byte_std)
        pixel_values = pixel_values.to(self.dtype)
        return pixel_values, tokens

    def _candidate(self, features):
        keys = set().union(*(feature.keys() for feature in features))
        if "vlm_content" not in keys:
            raise RuntimeError("GR00T inference batch has no vlm_content")

        contents = [
            feature["vlm_content"] for feature in features if "vlm_content" in feature
        ]
        pixel_values, tokens = self._image_and_tokens(contents)
        batch: dict[str, torch.Tensor] = dict(tokens)
        batch["pixel_values"] = pixel_values

        for key in keys:
            if key == "vlm_content":
                continue
            values = [feature[key] for feature in features if key in feature]
            stacked = self._stack(values)
            if stacked.is_floating_point():
                batch[key] = stacked.to(self.dtype).to(self.device)
            else:
                batch[key] = self._static_device(key, stacked)
        return self._batch_feature(batch)

    def _reference_on_device(self, reference):
        converted = {}
        for key, value in reference["inputs"].items():
            if value.is_floating_point():
                # Match policy._rec_to_dtype followed by model.prepare_input.
                converted[key] = value.to(self.dtype).to(self.device)
            else:
                converted[key] = value.to(self.device)
        return converted

    def _reject(self, reason: str) -> None:
        import sys

        self.rejected = True
        self.passed = False
        self.token_cache.clear()
        self.static_cache.clear()
        if self.policy.collate_fn is self:
            self.policy.collate_fn = self.original
        print(
            f"[{FAMILY}] rejected ({reason}); upstream CPU collate continues.",
            file=sys.stderr,
            flush=True,
        )

    def __call__(self, features):
        if self.rejected:
            return self.original(features)

        try:
            candidate = self._candidate(features)
        except Exception as error:  # noqa: BLE001 - upstream fallback is the contract
            self._reject(f"{type(error).__name__}: {error}")
            return self.original(features)

        if self.checks < self.SELF_CHECK_INPUTS:
            reference = self.original(features)
            reference_device = self._reference_on_device(reference)
            keys_match = set(reference_device) == set(candidate["inputs"])
            delta = (
                max(
                    (
                        _tensor_delta(reference_device[key], candidate["inputs"][key])
                        for key in reference_device
                    ),
                    default=0.0,
                )
                if keys_match
                else float("inf")
            )
            self.max_abs_delta = max(self.max_abs_delta, delta)
            self.checks += 1
            if delta != 0.0:
                self._reject(f"live self-check delta {delta:.3e}")
                return reference
            if self.checks == self.SELF_CHECK_INPUTS:
                self.passed = True
                print(
                    f"[{FAMILY}] bit-exact on {self.SELF_CHECK_INPUTS} live batches; "
                    "uint8 H2D + GPU normalize enabled."
                )

        self.calls += 1
        return candidate

    def close(self) -> None:
        if self.policy.collate_fn is self:
            self.policy.collate_fn = self.original
        self.token_cache.clear()
        self.static_cache.clear()
        if getattr(self.policy, "_instinctflash_gpu_collate", None) is self:
            delattr(self.policy, "_instinctflash_gpu_collate")


def install_gpu_collate(policy, *, device) -> GPUCollator:
    current = getattr(policy, "_instinctflash_gpu_collate", None)
    if current is not None:
        return current
    collator = GPUCollator(policy, device=torch.device(device))
    policy.collate_fn = collator
    policy._instinctflash_gpu_collate = collator
    return collator


__all__ = ["GPUCollator", "install_gpu_collate"]
