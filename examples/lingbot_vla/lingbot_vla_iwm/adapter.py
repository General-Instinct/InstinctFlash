"""Runtime adapter for ``robbyant/lingbot-vla-4b-posttrain-robotwin``.

Wraps the official serving class in-process — ``deploy.lingbot_vla_policy.LingbotVLAServer``
from the upstream checkout — exactly the way the H100 bench measured it (in-process ``infer``,
not the websocket hop; the two agree within 0.3%: 672.7 vs 670.9 ms stock). Preprocessing,
tokenization and action un-normalisation therefore stay byte-identical to upstream's server.

The T1 arm is :mod:`lingbot_vla_iwm.static_capture`: a replay-safe CUDA graph over the 10-step
denoise loop on static max-extent KV buffers — the 671 -> 185 ms row (54.7 -> 11.9 ms/step),
BITEXACT across six gate cases including a re-prefilled new prompt
(``examples/lingbot_vla/verify_static_capture.py``).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path

from instinctflash import AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec, PurityKey
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "lingbot_vla"
MODEL_ID = "robbyant/lingbot-vla-4b-posttrain-robotwin"
#: Where the upstream checkout is looked for when LINGBOT_VLA_ROOT is unset. The env var is the
#: contract; these are the documented conventions.
SOURCE_ROOT_CANDIDATES = (
    Path.home() / "lingbot-vla-repo",
    Path.home() / "lingbot-vla",
)
_CWD_LOCK = threading.RLock()


class LingBotVLA4BAdapter:
    """Three-camera RobotWin VLA: one Qwen2.5-VL prefill and ten action-flow steps."""

    def spec(self) -> AdapterSpec:
        # Prefix extent at the shipped robotwin configuration: three 224x224 images through
        # Qwen2.5-VL's ViT (patch 14, 2x2 merge -> 64 visual tokens each) plus language padded
        # to tokenizer_max_length=72 from the checkpoint's own config.json. The capture module
        # never trusts this number — it reads the true extent off the first prefill — but the
        # planner prices the prefix stream from it.
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=16_789_932_052,
            streams=(KVStreamSpec("prefix", tokens_per_frame=264, lifetime=KVLifetime.CHUNK),),
            phases=(
                PhaseSpec("prefix", nfe=1, writes=frozenset({"prefix"})),
                PhaseSpec("action", nfe=10, reads=frozenset({"prefix"}), truncatable=True,
                          min_nfe=1, depends_on=("prefix",)),
            ),
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            # Upstream already prefills once per chunk (fill_kv_cache=True on the first forward,
            # the K/V threaded through all ten denoise steps), so the purity is real but leaves
            # no work for a hoisting pass.
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
                "action_reply": "(use_length, 14); the shipped declaration serves use_length=25",
                "numeric_tier": "BITEXACT (static-KV capture, 6 gate cases all 0.0)",
            },
        )

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        ok, reason = imports_available(("torch", "torchvision", "transformers", "safetensors",
                                        "yaml", "lerobot"))
        if not ok:
            return ok, reason
        root = _source_root(required=False)
        if root is None or not (root / "deploy" / "lingbot_vla_policy.py").exists():
            return False, (f"LingBot-VLA source not found "
                           f"({'at ' + str(root) if root else 'no candidate exists'}); "
                           f"set LINGBOT_VLA_ROOT to the upstream checkout "
                           f"(github robbyant/lingbot-vla)")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from deploy.lingbot_vla_policy import LingbotVLAServer  # noqa: F401
        except Exception as error:  # noqa: BLE001 - import compatibility IS the host check
            return False, f"LingBot-VLA cannot import from {root}: {type(error).__name__}: {error}"
        return True, f"the model stack imports and LingBot-VLA source is at {root}"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("LingBot-VLA-4B inference requires CUDA")
        dev = str(device or "cuda")
        if not dev.startswith("cuda"):
            raise RuntimeError(f"LingBot-VLA-4B only supports a CUDA device, got {dev!r}")
        if ":" in dev:
            torch.cuda.set_device(int(dev.rsplit(":", 1)[1]))

        root = _source_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        extra = dict(checkpoint.execution.extra or {})
        # The Qwen2.5-VL base supplies tokenizer + image processor; upstream reads QWEN25_PATH.
        os.environ["QWEN25_PATH"] = os.environ.get("QWEN25_PATH") or str(
            extra.get("tokenizer_repo") or "Qwen/Qwen2.5-VL-3B-Instruct")
        model_path = _resolve_model_path(checkpoint)
        norm_path = _resolve_norm_stats(checkpoint, root)

        schedule = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        steps = int(schedule.get("action", 10))
        if steps < 1:
            raise ValueError(f"LingBot-VLA-4B action NFE must be positive, got {steps}")
        use_length = int(extra.get("use_length") or 25)

        from deploy.lingbot_vla_policy import LingbotVLAServer

        server = LingbotVLAServer(
            str(model_path), use_length=use_length, robot_norm_path=str(norm_path),
            num_denoising_step=steps,
        )
        driver = self.install(server, plan, device=dev)
        robot = str(extra.get("robot") or "robotwin")
        return _LingBotVLA4BLoop(server, root, robot=robot, driver=driver)

    def install(self, server, plan, *, device=None):
        """Install the static-KV denoise graph when the compiled plan applies graph_capture.

        The plan is READ, not decorative (same rule as GR00T and VLA-V2): a plan whose capture
        pass declined, or was excluded by the caller, must not be optimized around anyway.
        ``IFL_VLA4B_BACKEND=eager`` keeps the stock loop for A/B runs.
        """
        mode = os.environ.get("IFL_VLA4B_BACKEND", "static").strip().lower()
        if mode not in {"static", "eager"}:
            raise RuntimeError("IFL_VLA4B_BACKEND must be one of: static, eager")
        if mode == "eager":
            print("InstinctFlash LingBot-VLA-4B: using the upstream eager backend.")
            return None
        wanted = {
            getattr(result, "name", "")
            for result in getattr(plan, "results", ())
            if getattr(result, "applies", False)
        }
        if "graph_capture" not in wanted:
            print(
                "InstinctFlash LingBot-VLA-4B: the plan does not apply graph_capture, so the "
                "static-KV CUDA Graph backend is not installed; running the upstream path."
            )
            return None
        from .static_capture import install_static_capture

        driver = install_static_capture(server.vla.model)
        print("InstinctFlash LingBot-VLA-4B: static-KV CUDA Graph backend installed "
              "(bitexact-gated; see examples/lingbot_vla/verify_static_capture.py).")
        return driver


class _LingBotVLA4BLoop:
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
        # Upstream resolves configs/robot_configs/<robot>.yaml relative to its project root.
        # Scope the process-wide cwd change tightly and serialize it; inference itself uses no
        # relative paths.
        with _project_cwd(self._root):
            self._server.reset(self._robot)

    def predict(self, observation):
        import numpy as np

        obs = {}
        for key, value in observation.items():
            if hasattr(value, "detach"):                         # torch tensor from a caller
                value = value.detach().cpu().numpy()
            obs[key] = value
        prompt = str(obs.get("prompt") or obs.get("task") or self._prompt)
        if not prompt:
            raise ValueError("LingBot-VLA-4B requires a prompt (in reset() or predict())")
        obs["prompt"] = obs["task"] = prompt
        result = self._server.infer(obs)
        if "action" not in result:
            raise RuntimeError(
                f"upstream LingBot-VLA returned no 'action': {sorted(result.keys())}")
        return {"action": np.asarray(result["action"], dtype=np.float32)}

    @property
    def graph_stats(self) -> dict:
        d = self._driver
        return {
            "captured": bool(d and d._graph is not None),
            "replays": int(d.replays if d else 0),
        }

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        self._server = None


def _source_root(*, required: bool = True) -> "Path | None":
    env = os.environ.get("LINGBOT_VLA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in SOURCE_ROOT_CANDIDATES:
        if (candidate / "deploy" / "lingbot_vla_policy.py").exists():
            return candidate.resolve()
    if required:
        raise RuntimeError(
            "LingBot-VLA upstream source not found. Set LINGBOT_VLA_ROOT to the checkout "
            f"(searched: {[str(c) for c in SOURCE_ROOT_CANDIDATES]}).")
    return None


def _has_weights(path: Path) -> bool:
    return (next(path.glob("*.safetensors"), None) is not None
            and (path / "lingbotvla_cli.yaml").exists())


def _resolve_model_path(checkpoint) -> Path:
    """The checkpoint dir upstream's loader wants: flat safetensors + lingbotvla_cli.yaml."""
    root = Path(checkpoint.path)
    if _has_weights(root):
        return root
    pointer = (checkpoint.execution.extra or {}).get("base_weights")
    if pointer and Path(str(pointer)).exists():
        base = Path(str(pointer))
    elif pointer:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    if not _has_weights(base):
        raise RuntimeError(
            f"LingBot-VLA-4B weights not found under {base}: the upstream layout is flat "
            f"*.safetensors next to lingbotvla_cli.yaml (the published Hub release has both).")
    return base


def _resolve_norm_stats(checkpoint, source_root: Path) -> Path:
    """The action norm-stats file. A wrong or missing one silently denormalises into a space
    nobody can execute, so this is declared (execution.norm_stats) and verified, never guessed."""
    extra = dict(checkpoint.execution.extra or {})
    declared = str(extra.get("norm_stats") or "assets/norm_stats/robotwin_50.json")
    candidates = [Path(declared)] if Path(declared).is_absolute() else [
        Path(checkpoint.path) / declared,      # published beside the weights
        source_root / declared,                # shipped with the upstream checkout
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        f"{checkpoint.model_id}: norm stats {declared!r} not found "
        f"(tried {[str(c) for c in candidates]}). Declare execution.norm_stats as a path "
        f"relative to the package or the upstream checkout; the RoboTwin post-train uses "
        f"assets/norm_stats/robotwin_50.json.")


@contextlib.contextmanager
def _project_cwd(root: Path):
    with _CWD_LOCK:
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(previous)
