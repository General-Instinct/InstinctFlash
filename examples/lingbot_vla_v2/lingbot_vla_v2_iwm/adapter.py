"""Runtime adapter for ``robbyant/lingbot-vla-v2-6b-robotwin``.

The adapter preserves the upstream deploy server's preprocessing and action un-normalisation
contract.  Its default GPU preprocessing path keeps the upstream resize byte-for-byte and moves
the unchanged Qwen normalization/patchification work to CUDA.  Model execution uses dedicated
LingBot kernels and the replay-safe denoise executor installed by :mod:`static_capture`.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path

from instinctflash import AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec, PurityKey
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "lingbot_vla_v2"
MODEL_ID = "robbyant/lingbot-vla-v2-6b-robotwin"
#: Where the upstream checkout is looked for when LINGBOT_VLA_V2_ROOT is unset. There is no
#: universal default — the env var is the contract; these are the documented conventions.
SOURCE_ROOT_CANDIDATES = (
    Path.home() / "lingbot-vla-v2-repo",
    Path.home() / "lingbot-vla-v2",
)
_CWD_LOCK = threading.RLock()


class LingBotVLAV2Adapter:
    """Three-camera RobotWin VLA: one Qwen3-VL prefill and ten action-flow steps."""

    HOST_REQUIRES = (
        "torch", "torchvision", "transformers", "safetensors", "yaml", "flash_attn",
        "qwen_vl_utils", "lerobot",
    )

    def spec(self) -> AdapterSpec:
        # At the published 256x256 processor size each image contributes 64 visual tokens plus
        # Qwen's two vision-boundary tokens: 3*66.  The checkpoint pads language to 72 and appends
        # two 8-token task-query groups, giving a fixed prefix extent of 286.
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=25_503_630_044,
            streams=(KVStreamSpec("prefix", tokens_per_frame=286, lifetime=KVLifetime.CHUNK),),
            phases=(
                PhaseSpec("prefix", nfe=1, writes=frozenset({"prefix"})),
                PhaseSpec("action", nfe=10, reads=frozenset({"prefix"}), truncatable=True,
                          min_nfe=1, depends_on=("prefix",)),
            ),
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            purity=(PurityKey("prefix_kv", ("images", "state", "prompt"), KVLifetime.CHUNK,
                              already_hoisted=True),),
            observation=ObservationSpec(
                fields=(
                    ObservationField("observation.images.cam_high", (480, 640, 3), "uint8"),
                    ObservationField("observation.images.cam_left_wrist", (480, 640, 3), "uint8"),
                    ObservationField("observation.images.cam_right_wrist", (480, 640, 3), "uint8"),
                    ObservationField("observation.state", (14,), "float32"),
                ),
                history=1,
                batched=False,
                conditioning=("prompt",),
            ),
            notes={
                "family": "vla",
                "chunk_size": "50",
                "numeric_tier": "NUMERIC (upstream fused-MoE is nondeterministic)",
            },
        )

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        ok, reason = imports_available(self.HOST_REQUIRES)
        if not ok:
            return ok, reason
        root = _source_root(required=False)
        if root is None or not (root / "deploy" / "lingbot_vla_v2_policy.py").exists():
            return False, (f"LingBot-VLA-V2 source not found "
                           f"({'at ' + str(root) if root else 'no candidate exists'}); "
                           f"set LINGBOT_VLA_V2_ROOT to the upstream checkout")
        return True, f"the model stack imports and LingBot-VLA-V2 source is at {root}"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-VLA-V2 inference requires CUDA")
        dev = str(device or "cuda")
        if not dev.startswith("cuda"):
            raise RuntimeError(f"LingBot-VLA-V2 only supports a CUDA device, got {dev!r}")
        if ":" in dev:
            torch.cuda.set_device(int(dev.rsplit(":", 1)[1]))

        root = _source_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        extra = dict(checkpoint.execution.extra or {})
        qwen = os.environ.get("QWEN3VL_PATH") or str(
            extra.get("tokenizer_repo") or "Qwen/Qwen3-VL-4B-Instruct"
        )
        os.environ["QWEN3VL_PATH"] = qwen
        model_path = _resolve_model_path(checkpoint)

        mode = os.environ.get("IFL_VLA2_BACKEND", "static").strip().lower()
        if mode not in {"static", "compile", "eager"}:
            raise RuntimeError("IFL_VLA2_BACKEND must be one of: static, compile, eager")

        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        server = LingbotVLAv2Server(
            str(model_path), use_length=50, chunk_ret=True, use_bf16=True, use_fp32=False,
            use_compile=(mode == "compile"),
        )
        schedule = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        server.vla.model.config.num_steps = int(schedule.get("action", 10))

        driver = self.install(server, plan, mode=mode, device=dev)

        robot = str(extra.get("robot") or "robotwin")
        return _LingBotVLAV2Loop(server, root, robot=robot, driver=driver)

    def install(self, server, plan, *, mode="static", device=None):
        """Install the executor selected for this load, honoring the compiled plan.

        This public hook also tells ``InProcessBackend`` that the adapter acts on applicable plan
        results.  The plan is READ, not decorative: the CUDA-graph executors install only when the
        plan applies ``graph_capture`` (mirroring the GR00T adapter) — a plan whose capture pass
        declined, or was excluded by the caller, must not be optimized around anyway.
        """
        if mode == "static":
            wanted = {
                getattr(result, "name", "")
                for result in getattr(plan, "results", ())
                if getattr(result, "applies", False)
            }
            capture_planned = "graph_capture" in wanted
            if not capture_planned:
                print(
                    "InstinctFlash LingBot-VLA-V2: the plan does not apply graph_capture, so the "
                    "static-KV CUDA Graph backend is not installed; running the upstream path."
                )
            if _env_flag("IFL_VLA2_GPU_PREPROCESS", default=True):
                # FeatureTransform is created by the first server.reset(), which the Runtime loop
                # performs after installing compute backends. Defer this one transform-dependent
                # installer until that reset has completed.
                server._instinctflash_gpu_preprocess_pending = (
                    str(device or "cuda"),
                    os.environ.get("IFL_VLA2_GPU_PREPROCESS_MODE", "processor")
                    .strip()
                    .lower(),
                )
            # The Triton kernels DEFAULT OFF: their only accuracy evidence today is a 4-case
            # A100 gate against a null-derived threshold, weaker than the 6-case H100 protocol
            # behind the published row. Flip the default only after that gate passes on H100.
            all_kernels = _env_flag("IFL_VLA2_CUDA_KERNELS", default=False)
            if _env_flag("IFL_VLA2_MOE_KERNEL", default=all_kernels) \
                    and _triton_kernels_allowed("IFL_VLA2_MOE_KERNEL"):
                from .moe_kernel import install_lingbot_moe_kernel

                report = install_lingbot_moe_kernel(server.vla.model)
                server._instinctflash_moe_kernel = report
                print(
                    "InstinctFlash LingBot-VLA-V2: sparse-MoE CUDA kernel installed "
                    f"({report.layers} layers active, {report.converted_layers} converted)."
                )
            if _env_flag("IFL_VLA2_RMSNORM_KERNEL", default=all_kernels) \
                    and _triton_kernels_allowed("IFL_VLA2_RMSNORM_KERNEL"):
                from .rmsnorm_kernel import install_lingbot_rmsnorm_kernel

                report = install_lingbot_rmsnorm_kernel(server.vla.model)
                server._instinctflash_rmsnorm_kernel = report
                print(
                    "InstinctFlash LingBot-VLA-V2: fused RMSNorm CUDA kernel installed "
                    f"({report.modules} modules, hidden={report.hidden_size})."
                )
            driver = None
            if capture_planned:
                from .static_capture import install_static_capture

                driver = install_static_capture(server.vla.model)
                if _env_flag("IFL_VLA2_PREFIX_GRAPH", default=True):
                    from .prefix_capture import install_prefix_capture

                    prefix = install_prefix_capture(server.vla.model)
                    server._instinctflash_prefix_capture = prefix
                    print(
                        "InstinctFlash LingBot-VLA-V2: static vision/prefill CUDA Graph "
                        "backend installed."
                    )
                print("InstinctFlash LingBot-VLA-V2: static-KV CUDA Graph backend installed.")
            return driver
        if mode == "compile":
            print("InstinctFlash LingBot-VLA-V2: using upstream torch.compile backend.")
            return None
        print("InstinctFlash LingBot-VLA-V2: using upstream eager backend.")
        return None


class _LingBotVLAV2Loop:
    def __init__(self, server, source_root: Path, *, robot: str, driver=None):
        self._server = server
        self._root = source_root
        self._robot = robot
        self._driver = driver
        self._prompt = ""
        self.reset(robot=robot)

    def reset(self, **conditioning) -> None:
        self._prompt = str(conditioning.get("prompt") or "")
        self._robot = str(conditioning.get("robot") or self._robot)
        # Upstream resolves robot config and norm stats relative to its project root.  Scope the
        # process-wide cwd change tightly and serialize it; inference itself uses no relative paths.
        with _project_cwd(self._root):
            self._server.reset(self._robot)
        pending = getattr(
            self._server, "_instinctflash_gpu_preprocess_pending", None
        )
        if pending is not None:
            from .image_preprocess import install_lingbot_gpu_image_preprocess

            pending_device, pending_mode = pending
            report = install_lingbot_gpu_image_preprocess(
                self._server, device=pending_device, mode=pending_mode
            )
            self._server._instinctflash_gpu_preprocess = report
            delattr(self._server, "_instinctflash_gpu_preprocess_pending")
            print(
                "InstinctFlash LingBot-VLA-V2: GPU image preprocessing installed "
                f"({report.mode}, {report.camera_count} cameras, "
                f"{report.input_hw}->{report.output_hw})."
            )

    def predict(self, observation):
        obs = dict(observation)
        prompt = str(obs.get("prompt") or obs.get("task") or self._prompt)
        if not prompt:
            raise ValueError("LingBot-VLA-V2 requires a prompt (in reset() or predict())")
        obs["prompt"] = obs["task"] = prompt
        result = self._server.infer(obs)
        if "action" not in result:
            raise RuntimeError(f"upstream LingBot-VLA-V2 returned no 'action': {result.keys()}")
        return {"action": result["action"]}

    @property
    def graph_stats(self) -> dict:
        d = self._driver
        moe = getattr(self._server, "_instinctflash_moe_kernel", None)
        norm = getattr(self._server, "_instinctflash_rmsnorm_kernel", None)
        preprocess = getattr(self._server, "_instinctflash_gpu_preprocess", None)
        prefix = getattr(self._server, "_instinctflash_prefix_capture", None)
        return {
            "captured": bool(d and d.graph is not None),
            "replays": int(d.replays if d else 0),
            "cuda_kernels": bool(moe or norm),
            "moe_layers": int(moe.layers if moe else 0),
            "rmsnorm_modules": int(norm.modules if norm else 0),
            "gpu_image_preprocess": bool(preprocess),
            "vision_graph": bool(prefix and prefix.vision.graph is not None),
            "vision_replays": int(prefix.vision.replays if prefix else 0),
            "prefill_graph": bool(prefix and prefix.prefill.graph is not None),
            "prefill_replays": int(prefix.prefill.replays if prefix else 0),
        }

    def close(self) -> None:
        prefix = getattr(self._server, "_instinctflash_prefix_capture", None)
        if prefix is not None:
            prefix.close()
        if self._driver is not None:
            self._driver.close()
        moe = getattr(self._server, "_instinctflash_moe_kernel", None)
        if moe is not None:
            moe.close()
        preprocess = getattr(self._server, "_instinctflash_gpu_preprocess", None)
        if preprocess is not None:
            preprocess.close()
        self._server = None


def _source_root(*, required: bool = True) -> Path | None:
    env = os.environ.get("LINGBOT_VLA_V2_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in SOURCE_ROOT_CANDIDATES:
        if (candidate / "deploy" / "lingbot_vla_v2_policy.py").exists():
            return candidate.resolve()
    if required:
        raise RuntimeError(
            "LingBot-VLA-V2 upstream source not found. Set LINGBOT_VLA_V2_ROOT to the checkout "
            f"(searched: {[str(c) for c in SOURCE_ROOT_CANDIDATES]}).")
    return None


def _triton_kernels_allowed(flag_name: str) -> bool:
    """Refuse the Triton kernels on Thor (SM110) — measured dead, and worse than dead.

    Triton codegen fails on sm_110a with a PTXAS internal error (thor_column/vla2.json), and the
    vendor's MoE except-handler references an undefined ``logger``, so the failure surfaces as a
    NameError instead of a fallback; the RMSNorm patch has no try/except at all and dies at the
    first 768-wide norm. Refusing here is the only honest behaviour on that device.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (11, 0):
        raise RuntimeError(
            f"{flag_name} requests a Triton kernel on SM110 (Thor), where Triton codegen is "
            f"measured-dead (PTXAS internal error) and the vendor fallback path crashes. Use the "
            f"Thor engine arm (config 'lingbot_vla_v2', arch 'thor') instead.")
    return True


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean flag, got {value!r}")


def _resolve_model_path(checkpoint) -> Path:
    """Resolve flat, declared-view, and pointer-only package layouts."""
    root = Path(checkpoint.path)
    extra = dict(checkpoint.execution.extra or {})
    subdir = str(extra.get("checkpoint_subdir") or "")
    candidates = [root / subdir] if subdir else []
    candidates.append(root)
    for candidate in candidates:
        if ((candidate / "model.safetensors.index.json").exists()
                or next(candidate.glob("*.safetensors"), None) is not None):
            return candidate

    pointer = extra.get("base_weights")
    if pointer and Path(str(pointer)).exists():
        base = Path(str(pointer))
    elif pointer:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    candidate = base / subdir if subdir else base
    if not ((candidate / "model.safetensors.index.json").exists()
            or next(candidate.glob("*.safetensors"), None) is not None):
        raise RuntimeError(f"LingBot-VLA-V2 weights not found under {candidate}")
    return candidate


@contextlib.contextmanager
def _project_cwd(root: Path):
    with _CWD_LOCK:
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(previous)
