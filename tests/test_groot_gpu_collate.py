from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "examples" / "groot_n17"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from groot_n17_iwm.gpu_collate import GPUCollator, install_gpu_collate


class _ImageProcessor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    rescale_factor = 1.0 / 255.0
    do_rescale = True
    do_normalize = True

    def __call__(
        self,
        *,
        images,
        return_tensors,
        do_rescale=False,
        do_normalize=False,
    ):
        assert return_tensors == "pt"
        pixels = torch.stack(images).reshape(len(images), -1)
        if do_rescale or do_normalize:
            pixels = (pixels.float() - 127.5) / 127.5
        grid = torch.tensor([[1, 1, 1]] * len(images), dtype=torch.int64)
        return {"pixel_values": pixels, "image_grid_thw": grid}


class _Qwen:
    def __init__(self):
        self.image_processor = _ImageProcessor()
        self.calls = 0

    def __call__(
        self,
        *,
        text,
        images,
        return_tensors,
        padding,
        images_kwargs,
    ):
        self.calls += 1
        assert padding is True
        raw = self.image_processor(
            images=images,
            return_tensors=return_tensors,
            **images_kwargs,
        )
        return {
            "input_ids": torch.tensor([[11, 22, 33]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            **raw,
        }


class _UpstreamCollator:
    def __init__(self, *, pixel_bias=0.0):
        self.processor = _Qwen()
        self.pixel_bias = float(pixel_bias)
        self.calls = 0

    def __call__(self, features):
        self.calls += 1
        contents = [feature["vlm_content"] for feature in features]
        images = [image for content in contents for image in content["images"]]
        raw = self.processor.image_processor(
            images=images,
            return_tensors="pt",
            do_rescale=False,
            do_normalize=False,
        )
        pixels = (raw["pixel_values"].float() - 127.5) / 127.5
        pixels = pixels + self.pixel_bias
        return {
            "inputs": {
                "input_ids": torch.tensor([[11, 22, 33]], dtype=torch.int64),
                "attention_mask": torch.ones((1, 3), dtype=torch.int64),
                "pixel_values": pixels,
                "image_grid_thw": raw["image_grid_thw"],
                "state": torch.from_numpy(
                    np.stack([feature["state"] for feature in features])
                ),
                "embodiment_id": torch.from_numpy(
                    np.stack([feature["embodiment_id"] for feature in features])
                ),
            }
        }


def _feature(seed):
    generator = torch.Generator().manual_seed(seed)
    images = [
        torch.randint(0, 256, (3, 4, 5), dtype=torch.uint8, generator=generator)
        for _ in range(2)
    ]
    return {
        "vlm_content": {"text": "pick", "images": images},
        "state": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
        "embodiment_id": 24,
    }


def _reference_device(reference):
    return {
        key: (value.to(torch.bfloat16) if value.is_floating_point() else value)
        for key, value in reference["inputs"].items()
    }


def test_gpu_collate_six_live_batches_are_fieldwise_bitexact(monkeypatch):
    monkeypatch.setattr(GPUCollator, "_batch_feature", staticmethod(lambda values: {"inputs": values}))
    upstream = _UpstreamCollator()
    policy = SimpleNamespace(collate_fn=upstream)
    collator = install_gpu_collate(policy, device="cpu")

    for seed in range(6):
        feature = _feature(seed)
        got = policy.collate_fn([feature])
        expected = _reference_device(upstream([feature]))
        assert set(got["inputs"]) == set(expected)
        for key, value in expected.items():
            assert torch.equal(got["inputs"][key], value), key

    assert collator.passed is True
    assert collator.rejected is False
    assert collator.checks == 6
    assert collator.max_abs_delta == 0.0
    assert collator.qwen.calls == 1, "prompt/grid metadata should be cached"
    assert got["inputs"]["pixel_values"].dtype is torch.bfloat16

    collator.close()
    assert policy.collate_fn is upstream
    assert not hasattr(policy, "_instinctflash_gpu_collate")


def test_gpu_collate_mismatch_restores_upstream_immediately(monkeypatch):
    monkeypatch.setattr(GPUCollator, "_batch_feature", staticmethod(lambda values: {"inputs": values}))
    upstream = _UpstreamCollator(pixel_bias=0.25)
    policy = SimpleNamespace(collate_fn=upstream)
    collator = install_gpu_collate(policy, device="cpu")

    got = collator([_feature(0)])

    assert collator.rejected is True
    assert collator.passed is False
    assert collator.max_abs_delta > 0.0
    assert policy.collate_fn is upstream
    assert got["inputs"]["pixel_values"].dtype is torch.float32


def test_gpu_collate_rejects_nonuniform_qwen_normalization():
    upstream = _UpstreamCollator()
    upstream.processor.image_processor.image_mean = [0.4, 0.5, 0.6]
    policy = SimpleNamespace(collate_fn=upstream)

    with pytest.raises(ValueError, match="mean/std=0.5"):
        GPUCollator(policy, device=torch.device("cpu"))
