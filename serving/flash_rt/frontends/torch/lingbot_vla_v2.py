"""FlashRT Torch frontend for LingBot-VLA-V2.

This backend preserves the upstream BF16 processor/model contract and adds GPU image processing,
LingBot-specific Triton CUDA kernels, and static-KV CUDA Graph replay.  It is not a native FlashRT
FP8/CUTLASS port; that distinction keeps API support and kernel-certification claims separate.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import threading

import numpy as np

_CWD_LOCK = threading.RLock()
_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


class LingBotVLAV2TorchFrontend:
    """Official LingBot preprocessing plus a replay-safe static-KV denoise backend."""

    def __init__(
        self,
        checkpoint_dir: str,
        num_views: int = 3,
        use_cuda_graph: bool = True,
        # Triton kernels default OFF pending an H100 gate under the 6-case protocol; they are
        # refused outright on Thor SM110 (Triton measured-dead, vendor fallback crashes).
        use_cuda_kernels: bool = False,
        use_prefix_graph: bool = True,
        use_gpu_preprocess: bool = True,
        gpu_preprocess_mode: str = "processor",
        robot: str = "robotwin",
        source_root: str | None = None,
        qwen3vl_path: str | None = None,
    ):
        if int(num_views) != 3:
            raise ValueError("LingBot-VLA-V2 RobotWin requires exactly three camera views")
        self.num_views = 3
        self.robot = str(robot)
        resolved_root = source_root or os.environ.get("LINGBOT_VLA_V2_ROOT")
        if not resolved_root:
            for candidate in (pathlib.Path.home() / "lingbot-vla-v2-repo",
                              pathlib.Path.home() / "lingbot-vla-v2"):
                if (candidate / "deploy" / "lingbot_vla_v2_policy.py").exists():
                    resolved_root = candidate
                    break
        if not resolved_root:
            raise FileNotFoundError(
                "LingBot-VLA-V2 upstream source not found; pass source_root= or set "
                "LINGBOT_VLA_V2_ROOT to the checkout"
            )
        self.source_root = pathlib.Path(resolved_root).expanduser().resolve()
        source_file = self.source_root / "deploy" / "lingbot_vla_v2_policy.py"
        if not source_file.exists():
            raise FileNotFoundError(
                f"LingBot-VLA-V2 source not found at {source_file}; pass source_root= or set "
                "LINGBOT_VLA_V2_ROOT"
            )
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))

        self._checkpoint_path = str(_resolve_checkpoint(pathlib.Path(checkpoint_dir)))
        os.environ["QWEN3VL_PATH"] = str(
            qwen3vl_path or os.environ.get("QWEN3VL_PATH") or "Qwen/Qwen3-VL-4B-Instruct"
        )

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        self._server = LingbotVLAv2Server(
            self._checkpoint_path,
            use_length=50,
            chunk_ret=True,
            use_bf16=True,
            use_fp32=False,
            use_compile=False,
        )
        with _project_cwd(self.source_root):
            self._server.reset(self.robot)
        self._prompt: str | None = None
        self._graph_driver = None
        self._moe_kernel = None
        self._rmsnorm_kernel = None
        self._image_preprocess = None
        self._prefix_capture = None
        if use_cuda_graph or use_cuda_kernels or use_gpu_preprocess:
            # Imported per-feature: the Triton kernel modules import triton at module scope,
            # and pulling them in when only the graphs were requested couples unrelated
            # features to Triton availability.
            try:
                from lingbot_vla_v2_iwm.static_capture import install_static_capture
                from lingbot_vla_v2_iwm.prefix_capture import install_prefix_capture
                from lingbot_vla_v2_iwm.image_preprocess import (
                    install_lingbot_gpu_image_preprocess,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "LingBot FlashRT CUDA backends need the companion adapter package. "
                    "Install `examples/lingbot_vla_v2` into this environment."
                ) from exc
        if use_gpu_preprocess:
            self._image_preprocess = install_lingbot_gpu_image_preprocess(
                self._server, mode=str(gpu_preprocess_mode).strip().lower()
            )
        if use_cuda_kernels:
            import torch

            if torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0):
                raise RuntimeError(
                    "use_cuda_kernels=True requests Triton kernels on SM110 (Thor), where "
                    "Triton codegen is measured-dead (PTXAS internal error) and the vendor "
                    "fallback path crashes. Use the Thor engine arm "
                    "(config 'lingbot_vla_v2', arch 'thor') instead."
                )
            try:
                from lingbot_vla_v2_iwm.moe_kernel import install_lingbot_moe_kernel
                from lingbot_vla_v2_iwm.rmsnorm_kernel import install_lingbot_rmsnorm_kernel
            except ImportError as exc:
                raise RuntimeError(
                    "LingBot FlashRT CUDA kernels need the companion adapter package and "
                    "Triton. Install `examples/lingbot_vla_v2` into this environment."
                ) from exc
            try:
                self._moe_kernel = install_lingbot_moe_kernel(self._server.vla.model)
                self._rmsnorm_kernel = install_lingbot_rmsnorm_kernel(self._server.vla.model)
            except Exception:
                if self._moe_kernel is not None:
                    self._moe_kernel.close()
                raise
        if use_cuda_graph:
            self._graph_driver = install_static_capture(self._server.vla.model)
            if use_prefix_graph:
                self._prefix_capture = install_prefix_capture(self._server.vla.model)

    def set_prompt(self, prompt_text: str) -> None:
        self._prompt = str(prompt_text)

    def infer(self, observation) -> dict:
        prompt = str(observation.get("prompt") or observation.get("task") or self._prompt or "")
        if not prompt:
            raise ValueError("prompt is required on the first LingBot-VLA-V2 inference")
        state = observation.get("observation.state", observation.get("state"))
        if state is None:
            raise ValueError("LingBot-VLA-V2 requires a 14-value robot state")

        obs = {key: observation[key] for key in _CAMERA_KEYS if key in observation}
        if len(obs) != 3:
            images = observation.get("images")
            if images is None:
                images = [
                    observation.get("image"),
                    observation.get("wrist_image"),
                    observation.get("wrist_image_right"),
                ]
            if len(images) != 3 or any(image is None for image in images):
                raise ValueError("LingBot-VLA-V2 requires top, left-wrist, and right-wrist images")
            obs = dict(zip(_CAMERA_KEYS, images))
        obs["observation.state"] = np.asarray(state, dtype=np.float32)
        obs["prompt"] = obs["task"] = prompt
        result = self._server.infer(obs)
        return {"actions": np.asarray(result["action"], dtype=np.float32)}

    @property
    def backend_stats(self) -> dict:
        driver = self._graph_driver
        moe = self._moe_kernel
        norm = self._rmsnorm_kernel
        preprocess = self._image_preprocess
        prefix = self._prefix_capture
        return {
            "backend": (
                "lingbot_bf16_cuda_kernels_static_kv_cuda_graph"
                if driver and (moe or norm)
                else "upstream_bf16_static_kv_cuda_graph"
                if driver
                else "lingbot_bf16_cuda_kernels"
                if moe or norm
                else "upstream_bf16_eager"
            ),
            "captured": bool(driver and driver.graph is not None),
            "replays": int(driver.replays if driver else 0),
            "cuda_kernels": bool(moe or norm),
            "moe_layers": int(moe.layers if moe else 0),
            "rmsnorm_modules": int(norm.modules if norm else 0),
            "gpu_image_preprocess": bool(preprocess),
            "gpu_preprocess_mode": preprocess.mode if preprocess else None,
            "vision_graph": bool(prefix and prefix.vision.graph is not None),
            "vision_replays": int(prefix.vision.replays if prefix else 0),
            "prefill_graph": bool(prefix and prefix.prefill.graph is not None),
            "prefill_replays": int(prefix.prefill.replays if prefix else 0),
        }

    def close(self) -> None:
        if self._prefix_capture is not None:
            self._prefix_capture.close()
        if self._graph_driver is not None:
            self._graph_driver.close()
        if self._moe_kernel is not None:
            self._moe_kernel.close()
        if self._image_preprocess is not None:
            self._image_preprocess.close()
        self._server = None


def _resolve_checkpoint(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    nested = path / "checkpoints" / "global_step_50000" / "hf_ckpt"
    for candidate in (nested, path):
        if ((candidate / "model.safetensors.index.json").exists()
                or next(candidate.glob("*.safetensors"), None) is not None):
            return candidate
    raise FileNotFoundError(
        f"no LingBot-VLA-V2 safetensors found at {path} or {nested}"
    )


@contextlib.contextmanager
def _project_cwd(root: pathlib.Path):
    with _CWD_LOCK:
        previous = pathlib.Path.cwd()
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(previous)


__all__ = ["LingBotVLAV2TorchFrontend"]
