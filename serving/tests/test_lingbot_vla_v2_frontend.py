from __future__ import annotations

from unittest.mock import patch

import numpy as np

from flash_rt.api import load_model
from flash_rt.frontends.torch.lingbot_vla_v2 import LingBotVLAV2TorchFrontend


def test_frontend_translates_three_view_api_to_upstream_robotwin_keys():
    class FakeServer:
        def __init__(self):
            self.observation = None

        def infer(self, observation):
            self.observation = observation
            return {"action": np.ones((50, 14), dtype=np.float32)}

    frontend = LingBotVLAV2TorchFrontend.__new__(LingBotVLAV2TorchFrontend)
    frontend._server = FakeServer()
    frontend._prompt = "move the cup"
    frontend._graph_driver = None
    frontend._moe_kernel = None
    frontend._rmsnorm_kernel = None
    frontend._image_preprocess = None
    frontend._prefix_capture = None
    images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    result = frontend.infer({"images": images, "state": np.zeros(14, dtype=np.float32)})
    got = frontend._server.observation
    assert list(k for k in got if k.startswith("observation.images.")) == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    assert got["prompt"] == got["task"] == "move the cup"
    assert result["actions"].shape == (50, 14)


def test_public_loader_routes_lingbot_options_to_sm80_frontend():
    captured = {}

    class FakeFrontend:
        def __init__(self, checkpoint, num_views=3, use_cuda_graph=True, robot="robotwin",
                     source_root=None, qwen3vl_path=None, use_cuda_kernels=True,
                     use_prefix_graph=True,
                     use_gpu_preprocess=True, gpu_preprocess_mode="processor"):
            captured.update(locals())

        def set_prompt(self, prompt):
            pass

        def infer(self, observation):
            return {"actions": np.zeros((50, 14), dtype=np.float32)}

    with patch("flash_rt.hardware.detect_arch", return_value="cuda_sm80"), patch(
        "flash_rt.hardware.resolve_pipeline_class", return_value=FakeFrontend
    ):
        model = load_model(
            "/checkpoint", config="lingbot_vla_v2", framework="torch",
            use_cuda_graph=False, use_cuda_kernels=False, use_gpu_preprocess=False,
            use_prefix_graph=False, gpu_preprocess_mode="processor", robot="robotwin",
            source_root="/source",
            qwen3vl_path="Qwen/test",
        )
    assert model.framework == "torch"
    assert captured["num_views"] == 3
    assert captured["use_cuda_graph"] is False
    assert captured["use_cuda_kernels"] is False
    assert captured["use_prefix_graph"] is False
    assert captured["use_gpu_preprocess"] is False
    assert captured["gpu_preprocess_mode"] == "processor"
    assert captured["source_root"] == "/source"
    assert captured["qwen3vl_path"] == "Qwen/test"


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
                failures += 1
            else:
                print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
