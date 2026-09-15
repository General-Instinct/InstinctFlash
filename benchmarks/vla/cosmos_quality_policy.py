"""Strict, seeded Cosmos public Runtime adapter for isolated quality evaluation.

This policy implements ``infer`` for the existing openpi msgpack transport. It
does not implement a server, retries, action thresholding, or quality admission.
Each process owns one frozen matrix cell and an exclusively created output
directory. The simulator must explicitly reset and number every policy request.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import is_dataclass, replace
import functools
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import sys
import threading
import time


PROTOCOL = "cosmos-thor-quality-v1"
CHECKPOINTS = {
    "edge": (
        "nvidia/Cosmos3-Edge-Policy-DROID",
        "f9fddb427fd7cff0bef7c73791be36f0ab52d81e",
    ),
    "nano": (
        "nvidia/Cosmos3-Nano-Policy-DROID",
        "6706d7680581c255ff61e0f3bb49d90eac55c79e",
    ),
}
OBSERVATION_KEYS = {
    "observation/image",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
}
REQUEST_KEYS = {"episode_id", "request_id", "benchmark_identity_sha256"}


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path):
    path = Path(path).absolute()
    return {
        "path": str(path),
        "resolved_path": str(path.resolve(strict=True)),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _source_inventory():
    """Hash loaded dependencies and owned/vendor trees before lazy imports run."""
    from importlib import import_module

    paths = set()
    for package in ("instinctflash", "cosmos3_iwm", "cosmos_framework"):
        module = import_module(package)
        for directory in module.__path__:
            paths.update(
                p.resolve()
                for p in Path(directory).rglob("*")
                if p.is_file()
                and p.suffix in {".py", ".so", ".cu", ".cuh", ".cpp", ".h"}
            )
    prefixes = (
        "instinctflash",
        "flash_rt",
        "cosmos3_iwm",
        "cosmos_framework",
        "benchmarks.regression",
        "benchmarks.vla",
        "torch",
        "numpy",
        "transformers",
        "diffusers",
        "safetensors",
        "einops",
        "flash_attn",
    )
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith(prefixes) and filename and Path(filename).is_file():
            if Path(filename).suffix in {".py", ".so"}:
                paths.add(Path(filename).resolve())
    paths.add(Path(__file__).resolve())
    return {str(path): _sha(path) for path in sorted(paths)}


def _snapshot_directory(cell):
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            cell["model_id"], revision=cell["revision"], local_files_only=True
        )
    ).absolute()
    if snapshot.name != cell["revision"]:
        raise ValueError("Hub resolved a different checkpoint revision")
    return snapshot


def _checkpoint_inventory(cell, runtime=None):
    """Bind snapshot bytes before/after loading and verify the native view."""
    snapshot = _snapshot_directory(cell)
    native = (
        Path(_service(runtime).cfg.checkpoint_path) if runtime is not None else None
    )
    result = {}
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file() or path.suffix.lower() == ".md":
            continue
        relative = path.relative_to(snapshot)
        serving_path = native / relative if native is not None else path
        if serving_path.resolve(strict=True) != path.resolve(strict=True):
            raise ValueError(
                f"Native loading view does not match checkpoint artifact: {relative}"
            )
        result[str(relative)] = _file_identity(path)
    if not any(name.endswith(".safetensors") for name in result):
        raise ValueError(
            "Pinned released checkpoint requires actual safetensors weights"
        )
    return result


@contextmanager
def _observe_runtime_assets():
    """Observe and hash the actual external Wan load without replacing its data."""
    from cosmos_framework.utils.easy_io import easy_io

    original, owned = easy_io.load, "load" in vars(easy_io)
    records = []

    @functools.wraps(original)
    def observed(path, *args, **kwargs):
        if not isinstance(path, (str, Path)) or Path(path).name != "Wan2.2_VAE.pth":
            return original(path, *args, **kwargs)
        before = _file_identity(path)
        if (
            before["bytes"] != 2818839170
            or before["sha256"]
            != "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
        ):
            raise ValueError(
                "Actual external WanVAE differs from the measured runtime anchor"
            )
        result = original(path, *args, **kwargs)
        if _file_identity(path) != before:
            raise ValueError("WanVAE artifact changed during native loading")
        records.append(before)
        return result

    easy_io.load = observed
    try:
        yield records
        if len(records) != 1:
            raise ValueError("Require exactly one observed native external WanVAE load")
    finally:
        intact = easy_io.load is observed
        if owned:
            easy_io.load = original
        else:
            del easy_io.load
        if not intact:
            raise RuntimeError(
                "Native asset observer binding changed during construction"
            )


def _write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(path)


def _uint32(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value < 2**32
    ):
        raise ValueError(f"{name} must be an explicit uint32 integer")
    return int(value)


def request_seed(episode_id, benchmark_seed, request_id):
    """Protocol-reserved consecutive seeds, independent of the model or arm."""
    seed = _uint32(benchmark_seed, "benchmark_seed")
    index = _uint32(request_id, "request_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id must be a nonempty string")
    return _uint32(seed + index, "expanded request_seed")


expected_request_seed = request_seed


def frozen_cell(matrix_path, cell_id, *, matrix_sha256):
    """Validate the six explicitly requested cells without importing Torch."""
    matrix_path = Path(matrix_path)
    if _sha(matrix_path) != matrix_sha256:
        raise ValueError("Frozen user_e2e matrix SHA256 mismatch")
    matrix = json.loads(matrix_path.read_text())
    matches = [row for row in matrix["cells"] if row["id"] == cell_id]
    if len(matches) != 1:
        raise ValueError("Matrix requires exactly one matching cell")
    cell = copy.deepcopy(matches[0])
    family, arm = cell.get("family"), cell.get("arm")
    if family not in CHECKPOINTS or arm not in {
        "eager_native",
        "runtime_default",
        "runtime_selected",
    }:
        raise ValueError(
            "Quality policy permits only Edge/Nano eager/default/selected cells"
        )
    if (cell.get("model_id"), cell.get("revision")) != CHECKPOINTS[family]:
        raise ValueError("Checkpoint differs from the measured released policy")
    if cell_id != f"{family}-{arm}" or cell.get("action_shape") != [32, 8]:
        raise ValueError("Matrix identity or full raw action contract changed")
    expected_options = {} if arm == "eager_native" else {"device": "cuda:0"}
    if arm == "runtime_selected":
        expected_options["tier_ceiling"] = "numeric"
    if cell.get("expected_runtime_kwargs") != expected_options:
        raise ValueError(
            "Runtime kwargs differ from the measured native precision path"
        )
    schedule = {
        "sampler": "unipc",
        "steps": 4,
        "shift": 5.0,
        "guidance": 3.0,
        "nfe": {"prefix": 1, "action": 4},
    }
    if cell.get("effective_schedule") != schedule:
        raise ValueError("Require unchanged UniPC4 / CFG3 / shift5")
    environment = cell.get("expected_optimizer_environment")
    expected = (
        {}
        if arm != "runtime_selected"
        else {
            "IFL_COSMOS3_GEN_REGIONS": "1",
            "IFL_COSMOS3_TIMESTEP_CACHE": "1",
            "IFL_COSMOS3_SPLIT_PREFILL": "1",
            "IFL_COSMOS3_CONTIGUOUS_KV": "0",
            "IFL_BF16_LINEAR_RELU2": "1" if family == "edge" else "0",
        }
    )
    if family == "edge" and arm == "runtime_selected":
        library = (environment or {}).get("IFL_BF16_KERNEL_LIBRARY")
        if not isinstance(library, str) or not Path(library).is_absolute():
            raise ValueError("Edge requires its bound absolute BF16 library path")
        expected["IFL_BF16_KERNEL_LIBRARY"] = library
    if environment != expected:
        raise ValueError("Optimizer settings differ from the measured matrix cell")
    return cell


def _build_runtime(cell):
    if cell["arm"] == "eager_native":
        from instinctflash.descriptors.package import from_pretrained
        from benchmarks.regression.native_reference import build

        checkpoint = from_pretrained(cell["model_id"], revision=cell["revision"])
        return build(cell["family"], checkpoint, output_dir=None)
    from instinctflash import Runtime

    return Runtime.from_pretrained(
        cell["model_id"], revision=cell["revision"], **cell["expected_runtime_kwargs"]
    )


def _numeric_environment():
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (11, 0):
        raise ValueError("Quality policy requires one visible Jetson Thor GPU")
    result = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if any(result[name] for name in ("matmul_tf32", "cudnn_tf32", "cudnn_benchmark")):
        raise ValueError("Require measured TF32-off and cuDNN benchmark-off numerics")
    return result


def _service(runtime):
    from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop

    loop = getattr(runtime._backend, "_impl", None)
    if not isinstance(loop, CosmosDROIDLoop) or loop._fp8_receipt is not None:
        raise ValueError(
            "Require the actual native CosmosDROIDLoop behind public Runtime"
        )
    return loop._ensure_open()


def _observed_contract(runtime, cell):
    service = _service(runtime)
    cfg = service.cfg
    if not is_dataclass(cfg):
        raise ValueError(
            "Native service must expose its frozen dataclass configuration"
        )
    expected = {
        "domain_name": "droid_lerobot",
        "action_dim": 8,
        "action_chunk_size": 32,
        "conditioning_fps": 15.0,
        "num_steps": 4,
        "guidance": 3.0,
        "shift": 5.0,
        "history_length": 1,
        "use_state": True,
        "action_space": "joint_pos",
        "image_height": 540,
        "image_width": 640,
        "decode_video": False,
    }
    if any(getattr(cfg, key, None) != value for key, value in expected.items()):
        raise ValueError(
            "Actual native service differs from the full DROID output contract"
        )
    sampler = service.model.sampler
    sampler_type = type(sampler)
    if sampler_type.__name__ != "UniPCSampler" or not sampler_type.__module__.endswith(
        ".samplers.unipc"
    ):
        raise ValueError("Actual native sampler must be UniPCSampler")
    checkpoint = runtime._checkpoint
    if checkpoint.model_id != cell["model_id"]:
        raise ValueError("Loaded checkpoint model identity mismatch")
    # Declaration views use symlinked config/weights; resolve the actual config.
    config_path = (Path(checkpoint.path) / "config.json").resolve(strict=True)
    snapshot = _snapshot_directory(cell)
    if config_path != (snapshot / "config.json").resolve(strict=True):
        raise ValueError("Actual checkpoint config is not from the pinned revision")
    serving_config = (Path(cfg.checkpoint_path) / "config.json").resolve(strict=True)
    if serving_config != config_path:
        raise ValueError(
            "Runtime declaration and native service loaded different checkpoints"
        )
    execution = checkpoint.execution
    if execution.extra.get("format_prompt_as_json") is not (cell["family"] == "edge"):
        raise ValueError(
            "Checkpoint-specific Edge JSON / Nano plain prompt contract differs"
        )
    policy = runtime.execution_policy
    if cell["arm"] == "eager_native":
        if policy != {
            "reference": "upstream eager native policy; shared I/O translation only"
        }:
            raise ValueError(
                "Eager baseline must be the actual direct upstream Reference"
            )
        if any(name.startswith("_ifl_") for name in vars(service)):
            raise ValueError(
                "Direct eager reference must not install native optimization owners"
            )
    elif policy.get("precision") != "native" or policy.get("nfe") != {
        "prefix": 1,
        "action": 4,
    }:
        raise ValueError("Runtime precision or effective steps changed")
    if cell["arm"] == "runtime_selected":
        required = (
            "_ifl_numeric_attention",
            "_ifl_conditioning_cache",
            "_ifl_generation_regions",
            "_ifl_timestep_cache",
        )
        if any(getattr(service, name, None) is None for name in required):
            raise ValueError(
                "One of the measured NUMERIC optimizations was not installed"
            )
    return {
        "service": expected,
        "sampler": f"{sampler_type.__module__}.{sampler_type.__name__}",
        "format_prompt_as_json": cell["family"] == "edge",
        "checkpoint_config_sha256": _sha(config_path),
        "checkpoint_config_path": str(config_path),
        "execution_policy": copy.deepcopy(policy),
        "raw_action_shape": [32, 8],
        "action_postprocessing": "none; preserve raw public commands",
        "implementation": "direct_upstream_reference"
        if cell["arm"] == "eager_native"
        else "public_runtime",
    }


def _validate_observation(request):
    import numpy as np

    image = request["observation/image"]
    joint = request["observation/joint_position"]
    gripper = request["observation/gripper_position"]
    if (
        not isinstance(image, np.ndarray)
        or image.shape != (540, 640, 3)
        or image.dtype != np.uint8
    ):
        raise ValueError("Official Cosmos mosaic must be [540,640,3] uint8")
    for name, value, shapes in (
        ("joint_position", joint, {(7,), (1, 7)}),
        ("gripper_position", gripper, {(), (1,), (1, 1)}),
    ):
        array = np.asarray(value)
        if (
            array.shape not in shapes
            or array.dtype.kind != "f"
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                f"Require finite current measured {name}, with no extra history"
            )
    if not isinstance(request["prompt"], str) or not request["prompt"].strip():
        raise ValueError("Require an explicit nonempty original task prompt")
    return {key: copy.deepcopy(request[key]) for key in OBSERVATION_KEYS}


def _execution_stats(runtime, cell):
    """Reject disabled NUMERIC owners; counters are evidence, not task scores."""
    if cell["arm"] != "runtime_selected":
        return {"route": cell["arm"]}
    service = _service(runtime)
    result = {
        name: getattr(service, "_ifl_" + name).report()
        for name in (
            "numeric_attention",
            "conditioning_cache",
            "generation_regions",
            "timestep_cache",
        )
    }
    conditioning, regions = result["conditioning_cache"], result["generation_regions"]
    if result["numeric_attention"].get("backend") != "cudnn":
        raise ValueError("Selected NUMERIC attention owner did not report cuDNN")
    if (
        conditioning.get("disabled") is not False
        or conditioning.get("closed") is not False
        or conditioning.get("rejected")
    ):
        raise ValueError("Selected conditioning cache was disabled, closed or rejected")
    if (
        regions.get("closed") is not False
        or regions.get("rejected")
        or regions.get("compiled_calls", 0) < 1
    ):
        raise ValueError("Selected GEN regions failed or never executed")
    if not regions.get("regions") or any(
        r.get("state") != "ready" or r.get("error") is not None
        for r in regions["regions"]
    ):
        raise ValueError("Selected compiled GEN region is not ready")
    if result["timestep_cache"].get("closed") is not False:
        raise ValueError("Selected timestep cache was closed")
    return result


def seeded_predict(runtime, observation, seed):
    """One public prediction, with actual generation/UniPC/CFG calls observed.

    Restores the original configuration, ordinary request RNG and all temporary
    method bindings on success or failure. No model forward is replaced.
    """
    seed = _uint32(seed, "request_seed")
    service = _service(runtime)
    config, rng, model = service.cfg, service._rng, service.model
    rng_before = copy.deepcopy(rng.bit_generator.state)
    replacement = replace(config, seed=seed, deterministic_seed=True)
    sampler = model.sampler
    calls = {"generation": [], "sampler": [], "callbacks": 0, "branches": 0}
    bindings = []

    def patch(owner, name, wrapper):
        owned, original = name in vars(owner), getattr(owner, name)
        wrapped = wrapper(original)
        bindings.append((owner, name, owned, original, wrapped))
        setattr(owner, name, wrapped)

    def generation(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            expected = {"seed": [seed], "guidance": 3.0, "num_steps": 4, "shift": 5.0}
            if calls["generation"] or any(
                kwargs.get(k) != v for k, v in expected.items()
            ):
                raise ValueError("Actual generation seed or sampling contract differs")
            if "sampler" in kwargs and kwargs["sampler"] is not sampler:
                raise ValueError("Generation replaced the native UniPC sampler")
            calls["generation"].append(expected)
            return original(*args, **kwargs)

        return wrapped

    def sampling(original):
        @functools.wraps(original)
        def wrapped(callback, noise, *args, **kwargs):
            expected = {"seed": [seed], "num_steps": 4, "shift": 5.0}
            if (
                calls["sampler"]
                or args
                or any(kwargs.get(k) != v for k, v in expected.items())
            ):
                raise ValueError("Actual UniPC sampler arguments differ")
            calls["sampler"].append(expected)

            @functools.wraps(callback)
            def counted(*values, **named):
                calls["callbacks"] += 1
                return callback(*values, **named)

            return original(counted, noise, **kwargs)

        return wrapped

    def velocity(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            calls["branches"] += 1
            return original(*args, **kwargs)

        return wrapped

    try:
        service.cfg = replacement
        patch(model, "generate_samples_from_batch", generation)
        patch(sampler, "forward", sampling)
        patch(model, "_get_velocity", velocity)
        result = runtime.predict(observation)
        if (
            len(calls["generation"]) != 1
            or len(calls["sampler"]) != 1
            or (calls["callbacks"], calls["branches"]) != (4, 8)
        ):
            raise ValueError(
                "Require one complete public UniPC4 prediction and eight CFG branches"
            )
    finally:
        intact = service.cfg is replacement and all(
            getattr(owner, name) is wrapped for owner, name, _, _, wrapped in bindings
        )
        for owner, name, owned, original, _ in reversed(bindings):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        service.cfg = config
        rng_intact = service._rng is rng and _json_bytes(
            rng.bit_generator.state
        ) == _json_bytes(rng_before)
        service._rng = rng
        rng.bit_generator.state = rng_before
        if not intact or not rng_intact:
            raise RuntimeError(
                "Seeded quality prediction encountered concurrent binding/RNG mutation"
            )
    return result, {
        **calls,
        "request_seed": seed,
        "normal_seed_config_restored": True,
        "normal_request_rng_unchanged": True,
        "observers_restored": True,
    }


class CosmosQualityPolicy:
    """One isolated policy worker; invalid/repeated requests permanently stop it."""

    @classmethod
    def from_matrix(cls, matrix_path, cell_id, *, matrix_sha256, output_dir):
        cell = frozen_cell(matrix_path, cell_id, matrix_sha256=matrix_sha256)
        actual_env = {k: v for k, v in os.environ.items() if k.startswith("IFL_")}
        if actual_env != cell["expected_optimizer_environment"]:
            raise ValueError(
                "Worker IFL environment must exactly match the frozen matrix cell"
            )
        compiler_environment = os.environ.get("TORCHDYNAMO_DISABLE")
        if compiler_environment != ("1" if cell["arm"] == "eager_native" else "0"):
            raise ValueError(
                "TORCHDYNAMO_DISABLE must match eager/reference versus Runtime execution"
            )
        numeric = _numeric_environment()
        native = {}
        for key, value in actual_env.items():
            if key.endswith("_LIBRARY"):
                native[key] = {"path": value, "sha256": _sha(value)}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        runtime = None
        try:
            source_before = _source_inventory()
            checkpoint_before = _checkpoint_inventory(cell)
            with _observe_runtime_assets() as runtime_assets:
                runtime = _build_runtime(cell)
                # Runtime construction is lazy. Public reset creates the actual
                # service without generating any actions or borrowing a test input.
                runtime.reset(prompt="Initialize Cosmos quality worker")
            observed = _observed_contract(runtime, cell)
            checkpoint_artifacts = _checkpoint_inventory(cell, runtime)
            if checkpoint_artifacts != checkpoint_before:
                raise ValueError(
                    "Checkpoint artifact bytes changed during actual native loading"
                )
            source_inventory = _source_inventory()
            if any(
                source_inventory.get(path) != digest
                for path, digest in source_before.items()
            ):
                raise ValueError("Implementation sources changed during native loading")
            identity = {
                "protocol": PROTOCOL,
                "matrix_sha256": matrix_sha256,
                "cell_id": cell_id,
                "model_id": cell["model_id"],
                "revision": cell["revision"],
                "runtime_kwargs": cell["expected_runtime_kwargs"],
                "optimizer_environment": actual_env,
                "torchdynamo_disable": compiler_environment,
                "native_libraries": native,
                "checkpoint_artifacts": checkpoint_artifacts,
                "observed_external_assets": runtime_assets,
                "source_inventory": source_inventory,
                "source_inventory_scope": "All owned/core/Cosmos adapter/vendor source trees plus currently imported numerical dependency module files; not proof of executed lines",
                "numeric_environment": numeric,
                "observed_contract": observed,
                "policy_source_sha256": _sha(__file__),
                "task_quality_certified": False,
            }
            return cls(runtime, cell, identity, output_dir)
        except BaseException as error:
            _write_json(
                output_dir / "construction_failure.json",
                {"error": repr(error), "cell_id": cell_id},
            )
            if runtime is not None:
                runtime.close()
            raise

    def __init__(self, runtime, cell, identity, output_dir):
        self.runtime, self.cell, self.identity = (
            runtime,
            copy.deepcopy(cell),
            copy.deepcopy(identity),
        )
        self.output_dir = Path(output_dir)
        self.identity_sha256 = hashlib.sha256(_json_bytes(self.identity)).hexdigest()
        self.metadata = {
            "benchmark_identity": self.identity,
            "benchmark_identity_sha256": self.identity_sha256,
            "reset_extension": True,
            "request_ids_required": True,
            "action_shape": [32, 8],
            "action_postprocessing": "none",
        }
        self._episode = None
        self._episodes = set()
        self._ordinal = 0
        self._failed = self._closed = False
        self._lock = threading.Lock()
        _write_json(self.output_dir / "identity.json", self.metadata)

    def _check_live(self):
        if self._closed or self._failed:
            raise RuntimeError(
                "Quality worker is closed or failed; retries are forbidden"
            )
        if (
            _observed_contract(self.runtime, self.cell)
            != self.identity["observed_contract"]
        ):
            raise ValueError("Actual Runtime contract changed after worker admission")
        environment = {k: v for k, v in os.environ.items() if k.startswith("IFL_")}
        if (
            environment != self.identity["optimizer_environment"]
            or _numeric_environment() != self.identity["numeric_environment"]
            or os.environ.get("TORCHDYNAMO_DISABLE")
            != self.identity["torchdynamo_disable"]
        ):
            raise ValueError("Worker numerical environment changed after admission")

    def _failure(self, error):
        self._failed = True
        path = self.output_dir / "failure.json"
        if not path.exists():
            _write_json(
                path,
                {
                    "error": repr(error),
                    "episode": self._episode,
                    "next_output_ordinal": self._ordinal,
                    "benchmark_identity_sha256": self.identity_sha256,
                },
            )

    def reset_episode(self, episode_id, prompt, seed, *, max_policy_chunks):
        return self.infer(
            {
                "reset": True,
                "episode_id": episode_id,
                "prompt": prompt,
                "benchmark_seed": seed,
                "max_policy_chunks": max_policy_chunks,
                "benchmark_identity_sha256": self.identity_sha256,
            }
        )

    def infer(self, request):
        if not self._lock.acquire(blocking=False):
            self._failure(RuntimeError("Concurrent quality requests are forbidden"))
            raise RuntimeError("Concurrent quality requests are forbidden")
        try:
            self._check_live()
            if (
                not isinstance(request, dict)
                or request.get("benchmark_identity_sha256") != self.identity_sha256
            ):
                raise ValueError("Request must bind the exact admitted policy identity")
            if "reset" in request:
                return self._reset(request)
            return self._predict(request)
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            self._lock.release()

    def _reset(self, request):
        if (
            set(request)
            != {
                "reset",
                "episode_id",
                "prompt",
                "benchmark_seed",
                "max_policy_chunks",
                "benchmark_identity_sha256",
            }
            or request["reset"] is not True
        ):
            raise ValueError("Reset requires the exact explicit episode contract")
        episode_id, prompt = request["episode_id"], request["prompt"]
        seed = _uint32(request["benchmark_seed"], "benchmark_seed")
        bound = _uint32(request["max_policy_chunks"], "max_policy_chunks")
        if bound < 1 or seed + bound > 2**32:
            raise ValueError("Require a positive bounded uint32 request seed range")
        if (
            not isinstance(episode_id, str)
            or not episode_id.strip()
            or episode_id in self._episodes
        ):
            raise ValueError(
                "Episode identity must be new; duplicate resets/retries are forbidden"
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Reset requires the original nonempty task prompt")
        record = {
            "episode_id": episode_id,
            "prompt": prompt,
            "benchmark_seed": seed,
            "max_policy_chunks": bound,
            "next_request_id": 0,
            "episode_ordinal": len(self._episodes),
        }
        path = self.output_dir / f"episode_{len(self._episodes):06d}.json"
        _write_json(path.with_suffix(".intent.json"), request)
        self.runtime.reset(prompt=prompt)
        self._episodes.add(episode_id)
        self._episode = record
        ack = {
            "reset": True,
            "episode_id": episode_id,
            "benchmark_seed": seed,
            "max_policy_chunks": bound,
            "benchmark_identity_sha256": self.identity_sha256,
        }
        receipt_sha = _write_json(path, {**ack, "prompt": prompt})
        return {**ack, "receipt_sha256": receipt_sha}

    def _predict(self, request):
        import numpy as np
        from benchmarks.regression.user_e2e import request_hash

        if self._episode is None:
            raise ValueError("Prediction requires a seeded explicit episode reset")
        if set(request) != OBSERVATION_KEYS | REQUEST_KEYS:
            raise ValueError(
                "Prediction requires exact official Cosmos keys and request identity"
            )
        index = _uint32(request["request_id"], "request_id")
        if (
            request["episode_id"] != self._episode["episode_id"]
            or index != self._episode["next_request_id"]
        ):
            raise ValueError(
                "Duplicate, stale or reordered request; retries are forbidden"
            )
        if index >= self._episode["max_policy_chunks"]:
            raise ValueError("Request exceeds the frozen episode chunk/seed bound")
        if request["prompt"] != self._episode["prompt"]:
            raise ValueError("Prompt changed within the admitted simulator episode")
        observation = _validate_observation(request)
        seed = request_seed(
            self._episode["episode_id"], self._episode["benchmark_seed"], index
        )
        prefix = self.output_dir / f"request_{self._ordinal:06d}"
        before = request_hash(observation)
        record = {
            "episode_id": self._episode["episode_id"],
            "request_id": index,
            "request_seed": seed,
            "input_sha256": before,
            "benchmark_identity_sha256": self.identity_sha256,
        }
        _write_json(prefix.with_suffix(".intent.json"), record)
        started = time.perf_counter()
        result, execution = seeded_predict(self.runtime, observation, seed)
        seconds = time.perf_counter() - started
        if self._failed:
            raise RuntimeError("Concurrent failure invalidated this request")
        if request_hash(observation) != before:
            raise ValueError("Runtime mutated the bound raw public observation")
        action = result.get("action") if isinstance(result, dict) else None
        if (
            not isinstance(action, np.ndarray)
            or action.dtype != np.float32
            or action.shape != (32, 8)
            or not np.isfinite(action).all()
        ):
            raise ValueError("Require finite complete raw public action [32,8] float32")
        action = action.copy()
        actual_optimizations = _execution_stats(self.runtime, self.cell)
        current_sources = _source_inventory()
        if any(
            current_sources.get(path) != digest
            for path, digest in self.identity["source_inventory"].items()
        ):
            raise ValueError("Bound implementation source changed after admission")
        source_record = prefix.with_suffix(".sources.json")
        source_sha = _write_json(source_record, current_sources)
        arrays = prefix.with_suffix(".npz")
        with arrays.open("xb") as stream:
            np.savez_compressed(stream, action=action)
            stream.flush()
            os.fsync(stream.fileno())
        receipt = {
            **record,
            "status": "passed",
            "execution": execution,
            "actual_optimizations": actual_optimizations,
            "source_inventory_file": source_record.name,
            "source_inventory_sha256": source_sha,
            "action_file": arrays.name,
            "action_file_sha256": _sha(arrays),
            "action_sha256": request_hash(action),
            "action_shape": [32, 8],
            "action_dtype": "float32",
            "predict_host_seconds": seconds,
            "task_quality_certified": False,
        }
        receipt_sha = _write_json(prefix.with_suffix(".json"), receipt)
        self._episode["next_request_id"] += 1
        self._ordinal += 1
        return {
            "action": action,
            "episode_id": record["episode_id"],
            "request_id": index,
            "request_seed": seed,
            "benchmark_identity_sha256": self.identity_sha256,
            "receipt_sha256": receipt_sha,
        }

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self.runtime.close()
                finally:
                    _write_json(
                        self.output_dir / "closed.json",
                        {
                            "failed": self._failed,
                            "requests": self._ordinal,
                            "episodes": len(self._episodes),
                            "benchmark_identity_sha256": self.identity_sha256,
                        },
                    )
