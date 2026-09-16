"""The production InstinctFlash benchmark driver: real models, both arms, full contract.

One process serves ONE job (one arm of one pair) and exits. That is protocol, not convenience:
the V2 re-sweep measured that a second model built in the same process fails the capture
self-check and would time the loud fallback instead of the default arm, so per-arm process
isolation is the only honest way to run a pair (the runner already spawns one driver process
per job).

Arm dispatch is on the plan's preregistered ``arm.operating_point.name``:

``stock_upstream``
    The family's upstream serving surface, in process, eager — the stock arm of the family's
    ``reproduce_h100`` protocol. pi05 is LeRobot's processor pipeline + ``select_action`` with
    ``compile_model=False`` (the published rows' eager reference; the checkpoint ships
    ``compile_mode=max-autotune``, upstream's own compile default is a separate arm). GR00T is
    NVIDIA's ``Gr00tPolicy.get_action``. LingBot-VLA-4B/V2 are the official deploy servers'
    in-process ``infer`` (V2 with ``use_compile=False``, their eager reference).

``runtime_default``
    ``instinctflash.Runtime.from_pretrained`` with the arm's declared placement, precision and
    tier ceiling; omitted fields use the public Runtime defaults.

Suites: ``contract`` (four seeded cases, full ``action_values``) and ``latency`` (warm/timed
``full_policy_chunk`` samples). Open-loop needs the DROID graft and closed-loop needs simulator
infrastructure; both are refused loudly rather than approximated.

Action canonicalization (part of this driver's revision): actions are converted to float64,
dict outputs are flattened by sorted key, and the digest is SHA-256 over big-endian packed
doubles — the same convention as ``reference_driver.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vla.util import load_json, sha256_json, write_json_atomic  # noqa: E402


CONTRACT_TASKS = ("seeded_prompt_a", "seeded_prompt_b", "changed_prompt", "reset_replay")
LATENCY_TASK = "full_policy_chunk"
OPERATING_POINTS = ("stock_upstream", "runtime_default")

#: Fixed benchmark prompts per family. The verify/reproduce protocols' pairs where they exist;
#: prompt B always forces a prefix refill on prompt-cached paths.
PROMPTS = {
    "pi05": (
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "push the plate to the front of the stove",
    ),
    "groot_n17": (
        "pick up the object",
        "carefully pick up the leftmost small red block and place "
        "it inside the open drawer on the right side of the table",
    ),
    "lingbot_vla": (
        "Use the left arm to pick up the block and place it in the tray",
        "Stack the red bowl on top of the blue plate with the right arm",
    ),
    "lingbot_vla_v2": (
        "Use the left arm to pick up the block and place it in the tray",
        "Stack the red bowl on top of the blue plate with the right arm",
    ),
}

LINGBOT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


class DriverRefusal(RuntimeError):
    """This driver cannot honestly serve the request; the job must fail, never fabricate."""


# ----------------------------------------------------------------------------------------------
# identity


def driver_revision() -> str:
    """This checkout's git revision; measurements from a dirty tree are marked and rejected."""
    def _git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    if head is None:
        return f"file-sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return f"{head}-dirty" if dirty else head


def environment_fingerprint(backbone: str) -> str:
    import torch

    facts = {
        "backbone": backbone,
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return sha256_json(facts)


def resolve_snapshot(model_id: str, revision: str) -> Path:
    """The local Hub snapshot at exactly the locked revision, or a loud refusal.

    The runner sets HF_HUB_OFFLINE=1; this driver never downloads. For families whose adapter
    loads by repo id (pi05's ``base_weights``), the cache's ``refs/main`` must also point at
    the locked revision — otherwise the runtime arm would silently load different bytes than
    the revision this result claims.
    """
    if os.environ.get("HF_HUB_CACHE"):
        hub = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
        hub = hf_home / "hub"
    repo = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / revision
    if not snapshot.is_dir():
        raise DriverRefusal(f"model not cached at locked revision: {model_id}@{revision}")
    main_ref = repo / "refs" / "main"
    if main_ref.is_file() and main_ref.read_text().strip() != revision:
        raise DriverRefusal(
            f"{model_id}: the cache's refs/main points at {main_ref.read_text().strip()}, not the "
            f"locked revision {revision}; repo-id loads would serve different bytes than claimed"
        )
    return snapshot


def known_execution(model_id: str) -> dict:
    from instinctflash.descriptors.known import lookup

    doc = lookup(model_id)
    if doc is None:
        raise DriverRefusal(f"{model_id} has no known declaration; this driver serves built-ins")
    return dict(doc["execution"])


def _skip_transformers_mistral_hub_probe() -> None:
    """transformers 4.57.3 calls the Hub API even offline; none of these tokenizers is mistral."""
    try:
        import transformers.tokenization_utils_base as tub

        def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
            return tokenizer

        tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)
    except Exception:                              # noqa: BLE001 - older transformers: no probe
        pass


@contextlib.contextmanager
def _cwd(root: Path):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _require_root(env_name: str, probe: str) -> Path:
    root = os.environ.get(env_name)
    if not root:
        raise DriverRefusal(f"{env_name} must point at the upstream checkout")
    path = Path(root).expanduser().resolve()
    if not (path / probe).exists():
        raise DriverRefusal(f"{env_name}={path} is not the upstream checkout (missing {probe})")
    return path


# ----------------------------------------------------------------------------------------------
# canonical actions


def flatten_action(action) -> np.ndarray:
    """Canonical float64 flat vector. Dict outputs (GR00T) flatten by sorted key."""
    if isinstance(action, tuple):
        action = action[0]
    if isinstance(action, dict):
        parts = [np.asarray(_to_numpy(action[key]), dtype=np.float64).ravel()
                 for key in sorted(action)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    return np.asarray(_to_numpy(action), dtype=np.float64).ravel()


def _to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def action_digest(values: np.ndarray) -> str:
    packed = b"".join(struct.pack("!d", float(item)) for item in values.tolist())
    return hashlib.sha256(packed).hexdigest()


def seed_everything(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------
# family harnesses: observation builders + stock arms + runtime accessors


def lingbot_observation(seed: int, prompt: str) -> dict:
    """The verify_static_capture.py observation: 3 RoboTwin cameras + 14-dim state."""
    rng = np.random.default_rng(seed)
    obs = {key: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
           for key in LINGBOT_CAMERAS}
    obs["observation.state"] = rng.normal(0, 0.1, size=14).astype(np.float32)
    obs["task"] = obs["prompt"] = prompt
    return obs


def pi05_observation(seed: int, prompt: str, obs_features: dict) -> dict:
    """Seeded synthetic observation in the checkpoint's DECLARED geometry (unbatched;
    the upstream processor pipeline owns batching and normalization in both arms)."""
    rng = np.random.default_rng(seed)
    obs = {}
    for key, shape in obs_features.items():
        shape = tuple(int(item) for item in shape)
        if "image" in key or "camera" in key:
            obs[key] = rng.random(shape, dtype=np.float32)
        else:
            obs[key] = rng.normal(0, 0.1, size=shape).astype(np.float32)
    obs["prompt"] = prompt
    return obs


GROOT_STATE_DIMS = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
GROOT_VIDEO_BASES = ("exterior_image_1_left", "wrist_image_left")


def groot_state_by_base() -> dict:
    state = {}
    for base, dim in GROOT_STATE_DIMS.items():
        value = np.zeros((1, 1, dim), dtype=np.float32)
        if base == "eef_9d":
            value[0, 0, 3:9] = [1, 0, 0, 0, 1, 0]      # identity frame; zeros break the rot6d SVD
        state[base] = value
    return state


def groot_video(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {key: rng.integers(0, 256, size=(1, 2, 256, 256, 3), dtype=np.uint8)
            for key in GROOT_VIDEO_BASES}


class RuntimeArm:
    """The treatment: declared Runtime policy, predictions via the unified facade."""

    def __init__(self, request: dict):
        _skip_transformers_mistral_hub_probe()
        model_id = request["model"]["checkpoint"]["id"]
        revision = request["model"]["checkpoint"]["revision"]
        self._backbone = request["model"]["backbone"]
        snapshot = resolve_snapshot(model_id, revision)
        if self._backbone == "groot_n17":
            # the documented pin channel: the adapter prefers this path over refs/main
            os.environ["GR00T_N17_CHECKPOINT"] = str(snapshot)
        from instinctflash import Runtime

        self._runtime = Runtime.from_pretrained(model_id, revision=revision,
            placement=request["arm"]["operating_point"].get("placement", "auto"),
            precision=request["arm"]["operating_point"].get("precision", "native"),
            tier_ceiling=request["arm"]["operating_point"].get("tier_ceiling"))
        self._prompt = None

    def new_episode(self, prompt: str) -> None:
        self._runtime.reset(prompt=prompt)
        self._prompt = prompt

    def reset_chunk(self) -> None:
        if self._backbone == "pi05":                 # drop the buffered action chunk only
            self._runtime.reset(prompt=self._prompt)

    def predict(self, observation: dict) -> np.ndarray:
        out = self._runtime.predict(observation)
        key = "actions" if self._backbone == "groot_n17" else "action"
        return flatten_action(out[key])

    def close(self) -> None:
        self._runtime.close()


class Pi05Stock:
    """LeRobot's processor pipeline + select_action, eager (the published stock protocol)."""

    def __init__(self, request: dict):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        snapshot = resolve_snapshot(
            request["model"]["checkpoint"]["id"], request["model"]["checkpoint"]["revision"]
        )
        self._torch = torch
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        config = PI05Config.from_pretrained(snapshot)
        config.compile_model = False                 # the rows' eager reference arm
        self._policy = PI05Policy.from_pretrained(snapshot, config=config)
        self._policy.eval()
        self._policy.to(self._device)
        self._pre, self._post = make_pre_post_processors(
            self._policy.config, pretrained_path=snapshot,
            preprocessor_overrides={"device_processor": {"device": self._device}},
            postprocessor_overrides={"device_processor": {"device": self._device}},
        )
        self._prompt = ""

    def new_episode(self, prompt: str) -> None:
        self._prompt = prompt
        self._policy.reset()

    def reset_chunk(self) -> None:
        self._policy.reset()

    def predict(self, observation: dict) -> np.ndarray:
        torch = self._torch
        batch = {}
        for key, value in observation.items():
            if not key.startswith("observation."):
                continue
            tensor = torch.as_tensor(value)
            if tensor.dtype not in (torch.float32, torch.uint8):
                tensor = tensor.float()
            batch[key] = tensor.to(self._device)
        batch["task"] = str(observation.get("prompt") or self._prompt)
        with torch.no_grad():
            action = self._post(self._policy.select_action(self._pre(batch)))
        return flatten_action(action)

    def close(self) -> None:
        self._policy = None


class GrootStock:
    """NVIDIA's Gr00tPolicy.get_action, eager, DROID embodiment."""

    def __init__(self, request: dict):
        root = _require_root("GR00T_ROOT", "gr00t")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _skip_transformers_mistral_hub_probe()
        snapshot = resolve_snapshot(
            request["model"]["checkpoint"]["id"], request["model"]["checkpoint"]["revision"]
        )
        execution = known_execution(request["model"]["checkpoint"]["id"])
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self._policy = Gr00tPolicy(
            embodiment_tag=str(execution["embodiment_tag"]),
            model_path=str(snapshot), device="cuda:0",
        )
        self._config = self._policy.modality_configs
        self._state = groot_state_by_base()
        self._prompt = ""

    def new_episode(self, prompt: str) -> None:
        self._prompt = prompt

    def reset_chunk(self) -> None:
        pass

    def predict(self, observation: dict) -> np.ndarray:
        video = observation["video"]

        def rekey(values, keys):
            return {key: values[key.split(".")[-1]] for key in keys}

        obs = {
            "video": rekey(video, tuple(self._config["video"].modality_keys)),
            "state": rekey(self._state, tuple(self._config["state"].modality_keys)),
            "language": {key: [[self._prompt]] for key in self._config["language"].modality_keys},
        }
        return flatten_action(self._policy.get_action(obs))

    def close(self) -> None:
        self._policy = None


class LingbotVla4bStock:
    """The official deploy server's in-process infer (the 4B verify protocol's stock arm)."""

    def __init__(self, request: dict):
        self._root = _require_root("LINGBOT_VLA_ROOT", "deploy/lingbot_vla_policy.py")
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        _skip_transformers_mistral_hub_probe()
        execution = known_execution(request["model"]["checkpoint"]["id"])
        os.environ["QWEN25_PATH"] = os.environ.get("QWEN25_PATH") or str(
            execution.get("tokenizer_repo") or "Qwen/Qwen2.5-VL-3B-Instruct")
        snapshot = resolve_snapshot(
            request["model"]["checkpoint"]["id"], request["model"]["checkpoint"]["revision"]
        )
        norm_stats = self._root / str(execution["norm_stats"])
        if not norm_stats.exists():
            raise DriverRefusal(f"norm stats not found: {norm_stats}")
        from deploy.lingbot_vla_policy import LingbotVLAServer

        self._server = LingbotVLAServer(
            str(snapshot),
            use_length=int(execution["use_length"]),
            robot_norm_path=str(norm_stats),
            num_denoising_step=int(execution["nfe"]["action"]),
        )
        self._robot = str(execution.get("robot") or "robotwin")
        with _cwd(self._root):
            self._server.reset(self._robot)

    def new_episode(self, prompt: str) -> None:
        with _cwd(self._root):
            self._server.reset(self._robot)

    def reset_chunk(self) -> None:
        pass

    def predict(self, observation: dict) -> np.ndarray:
        return flatten_action(self._server.infer(dict(observation))["action"])

    def close(self) -> None:
        self._server = None


class LingbotVlaV2Stock:
    """The upstream V2 server, eager (use_compile=False), in process — the row's stock arm."""

    def __init__(self, request: dict):
        self._root = _require_root("LINGBOT_VLA_V2_ROOT", "deploy/lingbot_vla_v2_policy.py")
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))
        _skip_transformers_mistral_hub_probe()
        execution = known_execution(request["model"]["checkpoint"]["id"])
        os.environ["QWEN3VL_PATH"] = os.environ.get("QWEN3VL_PATH") or str(
            execution.get("tokenizer_repo") or "Qwen/Qwen3-VL-4B-Instruct")
        snapshot = resolve_snapshot(
            request["model"]["checkpoint"]["id"], request["model"]["checkpoint"]["revision"]
        )
        hf_ckpt = snapshot / str(execution["checkpoint_subdir"])
        if not hf_ckpt.is_dir():
            raise DriverRefusal(f"V2 checkpoint subdir not found: {hf_ckpt}")
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

        # use_length=50 is the reproduce_h100.py stock invocation (chunk 50, 10 steps, 3 cams).
        self._server = LingbotVLAv2Server(
            str(hf_ckpt), use_length=50, chunk_ret=True,
            use_bf16=True, use_fp32=False, use_compile=False,
        )
        self._robot = str(execution.get("robot") or "robotwin")
        with _cwd(self._root):
            self._server.reset(self._robot)

    def new_episode(self, prompt: str) -> None:
        with _cwd(self._root):
            self._server.reset(self._robot)

    def reset_chunk(self) -> None:
        pass

    def predict(self, observation: dict) -> np.ndarray:
        return flatten_action(self._server.infer(dict(observation))["action"])

    def close(self) -> None:
        self._server = None


STOCK_ARMS = {
    "pi05": Pi05Stock,
    "groot_n17": GrootStock,
    "lingbot_vla": LingbotVla4bStock,
    "lingbot_vla_v2": LingbotVlaV2Stock,
}


def make_observation(request: dict, seed: int, prompt: str) -> dict:
    backbone = request["model"]["backbone"]
    if backbone in {"lingbot_vla", "lingbot_vla_v2"}:
        return lingbot_observation(seed, prompt)
    if backbone == "pi05":
        execution = known_execution(request["model"]["checkpoint"]["id"])
        obs_features = execution.get("obs_features")
        if not isinstance(obs_features, dict) or not obs_features:
            raise DriverRefusal(
                f"{request['model']['checkpoint']['id']}: no declared obs_features; "
                f"a pi05 checkpoint states its own observation geometry"
            )
        return pi05_observation(seed, prompt, obs_features)
    if backbone == "groot_n17":
        video = groot_video(seed)
        nested = dict(video)
        nested.update({f"video.{key}": value for key, value in video.items()})
        state = groot_state_by_base()
        state_nested = dict(state)
        state_nested.update({f"state.{key}": value for key, value in state.items()})
        return {"video": nested, "state": state_nested}
    raise DriverRefusal(f"backbone {backbone!r} is not supported by this driver revision")


def build_arm(request: dict):
    point = request["arm"]["operating_point"]["name"]
    backbone = request["model"]["backbone"]
    if point not in OPERATING_POINTS:
        raise DriverRefusal(
            f"operating point {point!r} is not one this driver implements ({OPERATING_POINTS})"
        )
    if backbone not in STOCK_ARMS:
        raise DriverRefusal(f"backbone {backbone!r} is not supported by this driver revision")
    if point == "runtime_default":
        return RuntimeArm(request)
    return STOCK_ARMS[backbone](request)


# ----------------------------------------------------------------------------------------------
# suites


def run_contract(arm, request: dict, seed: int) -> dict:
    prompt_a, prompt_b = PROMPTS[request["model"]["backbone"]]
    task = request["task"]
    if task == "seeded_prompt_a":
        arm.new_episode(prompt_a)
        seed_everything(seed)
        values = arm.predict(make_observation(request, seed, prompt_a))
    elif task == "seeded_prompt_b":
        arm.new_episode(prompt_b)
        seed_everything(seed)
        values = arm.predict(make_observation(request, seed, prompt_b))
    elif task == "changed_prompt":
        # A predict under prompt A, then the SAME observation under prompt B in the same
        # episode: prompt-cached paths must refill their prefix, and the reported action is
        # the post-switch one.
        arm.new_episode(prompt_a)
        seed_everything(seed)
        arm.predict(make_observation(request, seed, prompt_a))
        arm.new_episode(prompt_b)
        seed_everything(seed)
        values = arm.predict(make_observation(request, seed, prompt_b))
    elif task == "reset_replay":
        # Predict, reset the episode, predict the identical observation again; report the
        # replay. Leaked per-episode state shows up as a digest change against seeded_prompt_a.
        arm.new_episode(prompt_a)
        seed_everything(seed)
        arm.predict(make_observation(request, seed, prompt_a))
        arm.new_episode(prompt_a)
        seed_everything(seed)
        values = arm.predict(make_observation(request, seed, prompt_a))
    else:
        raise DriverRefusal(f"unknown contract task {task!r}")
    return {
        "action_digest": action_digest(values),
        "action_values": [float(item) for item in values.tolist()],
        "finite": bool(np.isfinite(values).all()),
    }


def _cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:                            # CPU stub tests; real arms already need torch
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_latency(arm, request: dict, seed: int) -> dict:
    if request["task"] != LATENCY_TASK:
        raise DriverRefusal(f"unknown latency task {request['task']!r}")
    prompt_a, _ = PROMPTS[request["model"]["backbone"]]
    warmup = int(request["measurement"]["warmup"])
    iterations = int(request["measurement"]["iterations"])
    arm.new_episode(prompt_a)
    warm_observation = make_observation(request, seed + 1, prompt_a)
    timed_observation = make_observation(request, seed, prompt_a)
    for _ in range(warmup):
        arm.reset_chunk()
        seed_everything(seed)
        arm.predict(warm_observation)
    _cuda_synchronize()
    samples = []
    values = None
    for _ in range(iterations):
        arm.reset_chunk()
        seed_everything(seed)
        _cuda_synchronize()
        start = time.perf_counter()
        values = arm.predict(timed_observation)
        _cuda_synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_ms": samples,
        "action_digest": action_digest(values),
        "finite": bool(np.isfinite(values).all()),
    }


def run_job(job: dict) -> dict:
    request = job["request"]
    if request["schema_version"] != 1:
        raise DriverRefusal(f"unsupported request schema {request['schema_version']!r}")
    suite = request["suite"]
    if suite["kind"] not in {"contract", "latency"}:
        raise DriverRefusal(
            f"suite kind {suite['kind']!r} is not served by this driver: open_loop needs the "
            f"locked dataset graft and closed_loop needs simulator infrastructure; refusing "
            f"rather than approximating"
        )
    if suite["seed_strategy"] != "fixed":
        raise DriverRefusal("contract/latency suites preregister fixed seeds")
    seed = int(request["requested_seed"])
    backbone = request["model"]["backbone"]
    revision = driver_revision()
    if revision.endswith("-dirty"):
        raise DriverRefusal(
            f"driver checkout is dirty ({revision}); benchmark evidence must come from a "
            f"committed revision"
        )
    if revision != job["driver"]["revision"]:
        raise DriverRefusal(
            f"driver revision {revision} does not match the planned {job['driver']['revision']}"
        )
    arm = build_arm(request)
    try:
        if suite["kind"] == "contract":
            metrics = run_contract(arm, request, seed)
        else:
            metrics = run_latency(arm, request, seed)
    finally:
        arm.close()
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "request_sha256": job["request_sha256"],
        "status": "completed",
        "resolved_seed": seed,
        "metrics": metrics,
        "provenance": {
            "model_revision": request["model"]["checkpoint"]["revision"],
            "driver_revision": revision,
            "environment_fingerprint": environment_fingerprint(backbone),
            "synthetic": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    job = load_json(args.request)
    result = run_job(job)
    write_json_atomic(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
