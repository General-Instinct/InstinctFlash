"""Runtime adapter for the Cosmos3 action-policy family (Edge 3.86B / Nano 15.75B, DROID).

Wraps our patched cosmos-framework serving service in-process —
``cosmos_framework.scripts.action_policy_server_robotwin.RobotwinPolicyService`` — the same
policy pipeline the H100/Thor rows were measured through (the request is pushed through the
*training-time* ``ActionTransformPipeline``, so serve-time preprocessing stays byte-identical
to train-time). The published pairs used the pipeline arm's launch line
(``--num-steps 4 --guidance 1.0 --action-chunk-size 16 --action-dim 8 --domain-name
droid_lerobot --expected-image-height 540 --expected-image-width 640``); those values are the
DECLARATION's serving config here, never adapter constants.

The T1 CUDA-graphs arm (torch.compile mode="reduce-overhead" over the same weights: Edge
310.5 -> 185.8 ms vs pipeline 235.7 on H100) stays an OPTION — ``IFL_COSMOS3_CUDA_GRAPHS=1`` —
because inductor's cudagraph_trees asserts on a prompt change: the speedup holds for
single-prompt workloads only, and multi-prompt serving must stay on the pipeline arm. On Thor
the graphs arm is measured SLOWER (676 vs 660 ms); it is never a default anywhere.

One adapter, two declarations: Edge and Nano share every execution fact except size.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from instinctflash import AdapterSpec, GuidanceRule, PhaseSpec
from instinctflash.adapters.base import GuidanceMode, ObservationField, ObservationSpec

BACKBONE = "cosmos3_policy"
MODEL_ID = "nvidia/Cosmos3-Edge-Policy-DROID"
SERVER_MODULE = "cosmos_framework.scripts.action_policy_server_robotwin"

#: The declaration keys the serving config is built from. Each is a fact about how the
#: published rows were measured; a checkpoint that omits one is refused, not guessed at.
REQUIRED_SERVING_KEYS = (
    "domain_name", "action_dim", "action_chunk_size", "image_height", "image_width",
)


class Cosmos3PolicyAdapter:
    """Two-tower MoT action policy: one packed prefill, four UniPC denoise steps, no
    persistent KV — every request rebuilds its state, so shapes repeat across cycles."""

    CUDA_GRAPHS_ENV = "IFL_COSMOS3_CUDA_GRAPHS"

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id=MODEL_ID,
            param_bytes=7_574_066_016,       # Edge; the Nano declaration carries 31_499_049_824
            # No KV pool at all: the SequencePack is rebuilt per request and nothing is carried
            # between control cycles (upstream's notify_next_episode says so in as many words).
            streams=(),
            phases=(
                PhaseSpec("prefix", nfe=1, writes=frozenset()),
                PhaseSpec("action", nfe=4, truncatable=True, min_nfe=1,
                          depends_on=("prefix",)),
            ),
            # The model supports CFG, but the published operating point serves guidance=1.0 —
            # no negative branch, NFE == num_steps. A declaration raising guidance doubles the
            # network forwards; that is a different operating point with its own numbers.
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            observation=ObservationSpec(
                fields=(
                    ObservationField("image", (540, 640, 3), "uint8"),
                    ObservationField("state", (8,), "float32"),
                ),
                history=1,
                batched=False,
                conditioning=("prompt",),
            ),
            notes={
                "family": "action_policy",
                "action_reply": "(chunk, action_dim) = (16, 8) at the declared serving config",
                "sampler": "unipc, shift 5.0",
                "numeric_tier": ("NUMERIC (vs our own eager: <=1.6e-2 Edge / <=5e-2 Nano; "
                                 "null controls 0.0)"),
                "cuda_graphs": ("optional via IFL_COSMOS3_CUDA_GRAPHS=1; SINGLE-PROMPT ONLY "
                                "(inductor cudagraph_trees asserts on prompt change); measured "
                                "slower than the pipeline on Thor"),
            },
        )

    def observation_contract(self, checkpoint):
        """The request geometry FOR THIS CHECKPOINT, from its declaration.

        The image height/width guard is enforced by the service itself (a mismatched mosaic is
        refused at request time), so the contract shown to a caller must come from the same
        declared numbers — printing another embodiment's geometry would teach a wrong request.
        """
        import dataclasses

        extra = dict(checkpoint.execution.extra or {})
        missing = [k for k in REQUIRED_SERVING_KEYS if k not in extra]
        if missing:
            raise RuntimeError(_missing_serving_config_message(checkpoint, missing))
        fields = (
            ObservationField("image",
                             (int(extra["image_height"]), int(extra["image_width"]), 3),
                             "uint8"),
            ObservationField("state", (int(extra["action_dim"]),), "float32"),
        )
        return dataclasses.replace(self.spec().observation, fields=fields), \
            "the checkpoint's declared serving config (image_height/width, action_dim)"

    def can_host_in_process(self):
        from instinctflash.runtime.execution import imports_available

        ok, reason = imports_available(("torch", "numpy", "PIL", "pydantic"))
        if not ok:
            return ok, reason
        try:
            spec_found = importlib.util.find_spec(SERVER_MODULE)
        except ModuleNotFoundError:
            spec_found = None
        if spec_found is None:
            return False, (
                f"{SERVER_MODULE} is not importable. This adapter needs OUR patched "
                f"cosmos-framework checkout (the upstream release does not ship the robotwin "
                f"policy server); install it into this interpreter, e.g. "
                f"`uv sync --group=cu130-torch213` inside the patched cosmos-framework tree, "
                f"and run from that venv.")
        return True, "the model stack imports and the patched cosmos-framework server is present"

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos3 policy inference requires CUDA")
        dev = str(device or "cuda")
        if ":" in dev and dev.rsplit(":", 1)[1] not in ("", "0"):
            raise RuntimeError(
                f"the cosmos-framework stack pins itself to the process's first visible GPU "
                f"(its distributed init calls torch.cuda.set_device(0)); got device={dev!r}. "
                f"Select the GPU with CUDA_VISIBLE_DEVICES instead.")

        extra = dict(checkpoint.execution.extra or {})
        missing = [k for k in REQUIRED_SERVING_KEYS if k not in extra]
        if missing:
            raise RuntimeError(_missing_serving_config_message(checkpoint, missing))

        schedule = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        steps = int(schedule.get("action", 4))
        if steps < 1:
            raise ValueError(f"Cosmos3 action NFE must be positive, got {steps}")

        cuda_graphs = _env_flag(self.CUDA_GRAPHS_ENV,
                                default=bool(extra.get("cuda_graphs", False)))

        from cosmos_framework.scripts.action_policy_server_robotwin import (
            RobotwinPolicyService, RobotwinServerArgs,
        )

        args = RobotwinServerArgs(
            checkpoint_path=str(_resolve_model_path(checkpoint)),
            domain_name=str(extra["domain_name"]),
            action_dim=int(extra["action_dim"]),
            action_chunk_size=int(extra["action_chunk_size"]),
            expected_image_height=int(extra["image_height"]),
            expected_image_width=int(extra["image_width"]),
            num_steps=steps,
            guidance=float(extra.get("guidance", 1.0)),
            shift=float(extra.get("shift", 5.0)),
            seed=int(extra.get("seed", 0)),
            use_cuda_graphs=cuda_graphs,
            guardrails=False,        # gated repo; irrelevant to actions — same as every arm
        )
        service = RobotwinPolicyService(args)
        self.install(service, plan, cuda_graphs=cuda_graphs)
        return _Cosmos3PolicyLoop(service)

    def install(self, service, plan, *, cuda_graphs: bool = False):
        """Report what actually runs. The graphs arm is decided at LOAD (torch.compile inside
        OmniInference), so this cannot flip it — it can only tell the truth about it."""
        wanted = {
            getattr(result, "name", "")
            for result in getattr(plan, "results", ())
            if getattr(result, "applies", False)
        }
        if cuda_graphs:
            print(
                "InstinctFlash Cosmos3: CUDA-graphs arm ON (torch.compile reduce-overhead). "
                "SINGLE-PROMPT ONLY — inductor cudagraph_trees asserts on a prompt change; "
                "multi-prompt serving must use the pipeline arm (unset "
                f"{self.CUDA_GRAPHS_ENV}).")
            return ["graph_capture"]
        if "graph_capture" in wanted:
            print(
                "InstinctFlash Cosmos3: the plan applies graph_capture but the arm is opt-in "
                f"({self.CUDA_GRAPHS_ENV}=1) because it is single-prompt-only and measured "
                "slower than the pipeline on Thor. Serving the pipeline arm.")
        return []


class _Cosmos3PolicyLoop:
    """One control cycle = one policy request. Stateless across cycles by upstream design,
    so reset() only advances the episode counter and reseeds the request stream."""

    def __init__(self, service):
        self._service = service
        self._prompt = ""

    def reset(self, **conditioning) -> None:
        self._prompt = str(conditioning.get("prompt") or "")
        self._service.notify_next_episode()

    def predict(self, observation):
        import numpy as np

        obs = dict(observation)
        prompt = str(obs.get("prompt") or obs.get("task") or self._prompt)
        if not prompt:
            raise ValueError("Cosmos3 requires a prompt (in reset() or predict())")
        image = obs.get("image")
        if image is None:
            raise ValueError("Cosmos3 requires 'image': one RGB observation frame "
                             "(HxWx3 uint8, or an already-encoded base64 PNG string)")
        state = obs.get("state", obs.get("qpos14", obs.get("qpos")))
        if state is None:
            raise ValueError("Cosmos3 requires 'state': the absolute joint state vector")
        state = np.asarray(state, dtype=np.float32).reshape(-1)

        req = {
            "image": image if isinstance(image, str) else _encode_png_b64(image),
            "prompt": prompt,
            "state": [float(x) for x in state],
        }
        out = self._service.predict(req)
        action = np.asarray(out["action"], dtype=np.float32)
        return {"action": action, "timing": out.get("timing")}

    def close(self) -> None:
        self._service = None


def _encode_png_b64(image) -> str:
    """Lossless PNG round-trip — the same bytes-in-bytes-out channel every arm was measured
    over, so wrapping in-process changes nothing about what reaches the transform."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"'image' must be HxWx3 RGB, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"'image' must be uint8 pixels, got {array.dtype}")
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_model_path(checkpoint) -> Path:
    """The consolidated HF-layout checkpoint dir (config.json + model.safetensors.index.json).
    cosmos-framework wants a LOCAL ABSOLUTE path, so a Hub pointer is snapshotted first."""
    root = Path(checkpoint.path)
    if _is_cosmos_checkpoint(root):
        return root.resolve()
    pointer = (checkpoint.execution.extra or {}).get("base_weights")
    if pointer and Path(str(pointer)).exists():
        base = Path(str(pointer))
    elif pointer:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(str(pointer)))
    else:
        raise RuntimeError(f"{checkpoint.model_id}: no local weights and no base_weights pointer")
    if not _is_cosmos_checkpoint(base):
        raise RuntimeError(
            f"Cosmos3 checkpoint not found under {base}: expected the released HF layout "
            f"(config.json + model.safetensors.index.json + checkpoint.json).")
    return base.resolve()


def _is_cosmos_checkpoint(path: Path) -> bool:
    return ((path / "config.json").exists()
            and ((path / "model.safetensors.index.json").exists()
                 or next(path.glob("*.safetensors"), None) is not None))


def _missing_serving_config_message(checkpoint, missing) -> str:
    name = getattr(checkpoint.execution, "model_id", "") or getattr(checkpoint, "path", "")
    return (
        f"{name}: the declaration is missing "
        f"{missing} from its execution block. The Cosmos3 serving config is a set of "
        f"measured facts, not defaults — the published DROID rows used domain_name="
        f"'droid_lerobot', action_dim=8, action_chunk_size=16, image 540x640, num_steps 4, "
        f"guidance 1.0 — and a guessed value silently skews serve-time preprocessing away "
        f"from training. Declare them in the checkpoint's instinctflash.json.")


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
