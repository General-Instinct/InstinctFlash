"""CPU protocol checks: public calls, native seed binding and immutable receipts."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from benchmarks.vla import cosmos_quality_policy as quality
from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop


@dataclass(frozen=True)
class Config:
    checkpoint_path: str
    seed: int = 17
    deterministic_seed: bool = False
    domain_name: str = "droid_lerobot"
    action_dim: int = 8
    action_chunk_size: int = 32
    conditioning_fps: float = 15.0
    num_steps: int = 4
    guidance: float = 3.0
    shift: float = 5.0
    history_length: int = 1
    use_state: bool = True
    action_space: str = "joint_pos"
    image_height: int = 540
    image_width: int = 640
    decode_video: bool = False


class UniPCSampler:
    def forward(self, callback, noise, *, seed, num_steps, shift):
        for _ in range(num_steps):
            callback(noise)
        return noise


UniPCSampler.__module__ = "test_vendor.samplers.unipc"


class Model:
    def __init__(self):
        self.sampler = UniPCSampler()
        self.fail = False
        self.bad_branches = False

    def _get_velocity(self, value):
        return value

    def generate_samples_from_batch(self, batch, *, seed, guidance, num_steps, shift):
        if self.fail:
            raise RuntimeError("native model failure")

        def callback(noise):
            self._get_velocity(noise)
            if not self.bad_branches:
                self._get_velocity(noise)
            return noise

        self.sampler.forward(
            callback, None, seed=seed, num_steps=num_steps, shift=shift
        )
        values = np.random.default_rng(seed[0]).normal(size=(32, 8)).astype(np.float32)
        return {"action": values}


class Service:
    def __init__(self, path):
        self.cfg = Config(str(path))
        self.model = Model()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._lock = threading.Lock()

    def infer(self, observation):
        seed = (
            self.cfg.seed
            if self.cfg.deterministic_seed
            else int(self._rng.integers(0, 2**31))
        )
        return self.model.generate_samples_from_batch(
            observation,
            seed=[seed],
            guidance=self.cfg.guidance,
            num_steps=self.cfg.num_steps,
            shift=self.cfg.shift,
        )


class Runtime:
    def __init__(self, cell, snapshot):
        self.service = Service(snapshot)
        self.loop = CosmosDROIDLoop(self.service)
        self._backend = SimpleNamespace(_impl=self.loop)
        self._checkpoint = SimpleNamespace(
            path=str(snapshot),
            model_id=cell["model_id"],
            execution=SimpleNamespace(
                extra={"format_prompt_as_json": cell["family"] == "edge"}
            ),
        )
        self.execution_policy = {
            "precision": "native",
            "nfe": {"prefix": 1, "action": 4},
        }
        if cell["arm"] == "eager_native":
            self.execution_policy = {
                "reference": "upstream eager native policy; shared I/O translation only"
            }
        self.public_resets = []
        self.public_predicts = []
        self.closed = False
        self.output_change = None
        if cell["arm"] == "runtime_selected":
            reports = {
                "_ifl_numeric_attention": {"backend": "cudnn"},
                "_ifl_conditioning_cache": {
                    "disabled": False,
                    "closed": False,
                    "rejected": [],
                },
                "_ifl_generation_regions": {
                    "closed": False,
                    "rejected": [],
                    "compiled_calls": 1,
                    "regions": [{"state": "ready", "error": None}],
                },
                "_ifl_timestep_cache": {"closed": False},
            }
            for name in (
                "_ifl_numeric_attention",
                "_ifl_conditioning_cache",
                "_ifl_generation_regions",
                "_ifl_timestep_cache",
            ):
                setattr(
                    self.service,
                    name,
                    SimpleNamespace(
                        close=lambda: None, report=lambda name=name: reports[name]
                    ),
                )

    def reset(self, **kwargs):
        self.public_resets.append(kwargs)
        return self.loop.reset(**kwargs)

    def predict(self, observation):
        self.public_predicts.append(observation)
        result = self.loop.predict(observation)
        return self.output_change(result) if self.output_change else result

    def close(self):
        self.closed = True
        self.loop.close()


@pytest.fixture
def build(tmp_path, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("IFL_"):
            monkeypatch.delenv(key)
    source = (
        Path(__file__).resolve().parents[1]
        / "eval/user_e2e_2026-09-14/matrix_final.json"
    )
    original = json.loads(source.read_text())
    calls = []
    source_fixture = tmp_path / "native_source.py"
    source_fixture.write_text("# frozen CPU source fixture\n")

    @contextmanager
    def observed_assets():
        yield [{"path": "test-only-Wan.pth", "sha256": "test-observed-asset"}]

    monkeypatch.setattr(quality, "_observe_runtime_assets", observed_assets)
    monkeypatch.setattr(
        quality, "_snapshot_directory", lambda c: tmp_path / c["revision"]
    )
    monkeypatch.setattr(
        quality,
        "_source_inventory",
        lambda: {str(source_fixture): quality._sha(source_fixture)},
    )

    def create(family="edge", arm="runtime_default", *, transform=None):
        cell = next(
            row.copy() for row in original["cells"] if row["id"] == f"{family}-{arm}"
        )
        cell["expected_optimizer_environment"] = dict(
            cell["expected_optimizer_environment"]
        )
        library = tmp_path / "libbf16.so"
        if "IFL_BF16_KERNEL_LIBRARY" in cell["expected_optimizer_environment"]:
            library.write_bytes(b"CPU fixture native library")
            cell["expected_optimizer_environment"]["IFL_BF16_KERNEL_LIBRARY"] = str(
                library
            )
        path = tmp_path / f"matrix_{len(calls)}.json"
        path.write_text(json.dumps({"cells": [cell]}))
        snapshot = tmp_path / cell["revision"]
        snapshot.mkdir(exist_ok=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"CPU test weight identity")
        runtime = Runtime(cell, snapshot)
        if transform:
            transform(runtime)
        monkeypatch.setattr(
            quality,
            "_numeric_environment",
            lambda: {
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "cudnn_benchmark": False,
            },
        )
        monkeypatch.setattr(quality, "_build_runtime", lambda c: runtime)
        monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1" if arm == "eager_native" else "0")
        for key in list(__import__("os").environ):
            if key.startswith("IFL_"):
                monkeypatch.delenv(key)
        for key, value in cell["expected_optimizer_environment"].items():
            monkeypatch.setenv(key, value)
        directory = tmp_path / f"receipts_{len(calls)}"
        calls.append((cell, runtime, path, directory))
        policy = quality.CosmosQualityPolicy.from_matrix(
            path, cell["id"], matrix_sha256=quality._sha(path), output_dir=directory
        )
        return policy, runtime

    return create


def observation(policy, request_id=0, *, episode_id="episode-1"):
    return {
        "observation/image": np.zeros((540, 640, 3), np.uint8),
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.37], np.float32),
        "prompt": "put the cube in the bowl",
        "episode_id": episode_id,
        "request_id": request_id,
        "benchmark_identity_sha256": policy.identity_sha256,
    }


def reset(policy, *, episode_id="episode-1", seed=23, bound=3):
    return policy.reset_episode(
        episode_id, "put the cube in the bowl", seed, max_policy_chunks=bound
    )


@pytest.mark.parametrize(
    "family,arm",
    [
        (f, a)
        for f in ("edge", "nano")
        for a in ("eager_native", "runtime_default", "runtime_selected")
    ],
)
def test_exact_public_runtime_calls_raw_actions_and_receipts(build, family, arm):
    policy, runtime = build(family, arm)
    ack = reset(policy)
    assert ack["benchmark_seed"] == 23 and ack["max_policy_chunks"] == 3
    state = runtime.service._rng.bit_generator.state.copy()
    config = runtime.service.cfg
    response = policy.infer(observation(policy))
    expected = np.random.default_rng(23).normal(size=(32, 8)).astype(np.float32)
    np.testing.assert_array_equal(response["action"], expected)
    assert not set(np.unique(response["action"][:, -1])).issubset({0, 1})
    assert runtime.public_resets == [
        {"prompt": "Initialize Cosmos quality worker"},
        {"prompt": "put the cube in the bowl"},
    ]
    assert len(runtime.public_predicts) == 1
    assert set(runtime.public_predicts[0]) == quality.OBSERVATION_KEYS
    assert (
        runtime.service.cfg is config
        and runtime.service._rng.bit_generator.state == state
    )
    receipt = policy.output_dir / "request_000000.json"
    assert response["receipt_sha256"] == quality._sha(receipt)
    record = json.loads(receipt.read_text())
    assert record["source_inventory_sha256"] == quality._sha(
        policy.output_dir / record["source_inventory_file"]
    )
    assert (
        record["execution"]["branches"] == 8 and record["execution"]["callbacks"] == 4
    )
    assert record["execution"]["generation"][0]["seed"] == [23]
    with np.load(policy.output_dir / record["action_file"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["action"], expected)
    second = policy.infer(observation(policy, 1))
    assert second["request_seed"] == 24
    assert not np.array_equal(response["action"], second["action"])
    assert not vars(
        runtime.service.model.sampler
    )  # temporary instance forward restored
    assert "generate_samples_from_batch" not in vars(runtime.service.model)
    policy.close()
    assert runtime.closed


def test_pairing_is_model_arm_independent_and_episode_range_bounded(build):
    left, _ = build("edge")
    right, _ = build("nano")
    reset(left)
    reset(right)
    np.testing.assert_array_equal(
        left.infer(observation(left))["action"],
        right.infer(observation(right))["action"],
    )
    assert quality.expected_request_seed("any episode", 100, 2) == 102
    with pytest.raises(ValueError, match="uint32"):
        quality.request_seed("e", 2**32 - 1, 1)


@pytest.mark.parametrize("value", [True, -1, 2**32, 1.5, "12"])
def test_invalid_request_seed_fails_before_runtime(value):
    with pytest.raises(ValueError, match="uint32"):
        quality.seeded_predict(None, {}, value)


@pytest.mark.parametrize(
    "mode",
    ["duplicate", "reorder", "wrong_episode", "prompt", "extra", "identity", "bound"],
)
def test_bad_wire_requests_poison_worker_without_native_retry(build, mode):
    policy, runtime = build()
    reset(policy, bound=1 if mode == "bound" else 3)
    policy.infer(observation(policy))
    request = observation(policy, 1)
    if mode == "duplicate":
        request["request_id"] = 0
    if mode == "reorder":
        request["request_id"] = 2
    if mode == "wrong_episode":
        request["episode_id"] = "other"
    if mode == "prompt":
        request["prompt"] = "different instruction"
    if mode == "extra":
        request["executed_action"] = np.zeros((32, 8), np.float32)
    if mode == "identity":
        request["benchmark_identity_sha256"] = "wrong"
    with pytest.raises(ValueError):
        policy.infer(request)
    with pytest.raises(RuntimeError, match="retries"):
        policy.infer(observation(policy, 1))
    assert len(runtime.public_predicts) == 1
    assert (policy.output_dir / "failure.json").is_file()


def test_duplicate_episode_and_predict_before_reset_are_rejected(build):
    policy, runtime = build()
    with pytest.raises(ValueError, match="reset"):
        policy.infer(observation(policy))
    assert not runtime.public_predicts
    other, _ = build()
    reset(other)
    with pytest.raises(ValueError, match="duplicate"):
        reset(other)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update({"observation/image": np.zeros((360, 640, 3), np.uint8)}),
        lambda r: r.update({"observation/image": np.zeros((540, 640, 3), np.float32)}),
        lambda r: r.update(
            {"observation/joint_position": np.zeros((2, 7), np.float32)}
        ),
        lambda r: r.update(
            {"observation/gripper_position": np.array([np.nan], np.float32)}
        ),
    ],
)
def test_invalid_public_observations_never_reach_model(build, mutate):
    policy, runtime = build()
    reset(policy)
    request = observation(policy)
    mutate(request)
    with pytest.raises(ValueError):
        policy.infer(request)
    assert not runtime.public_predicts


@pytest.mark.parametrize(
    "output",
    [
        lambda r: {"action": r["action"][:16]},
        lambda r: {"action": r["action"].astype(np.float64)},
        lambda r: {"action": np.full((32, 8), np.nan, np.float32)},
        lambda r: {"actions": r["action"]},
    ],
)
def test_full_finite_raw_output_required_before_any_reply(build, output):
    policy, runtime = build()
    reset(policy)
    runtime.output_change = output
    with pytest.raises(ValueError, match="complete raw"):
        policy.infer(observation(policy))
    assert (policy.output_dir / "request_000000.intent.json").is_file()
    assert not (policy.output_dir / "request_000000.json").exists()


@pytest.mark.parametrize(
    "reason", ["native_fail", "branch_count", "wrong_seed", "rng_consumption"]
)
def test_failed_native_calls_restore_bindings_and_rng(build, reason):
    policy, runtime = build()
    reset(policy)
    service = runtime.service
    if reason == "native_fail":
        service.model.fail = True
    if reason == "branch_count":
        service.model.bad_branches = True
    if reason == "wrong_seed":

        def wrong(observation):
            return service.model.generate_samples_from_batch(
                observation, seed=[5], guidance=3.0, num_steps=4, shift=5.0
            )

        runtime.predict = wrong
    if reason == "rng_consumption":
        predict = runtime.predict

        def consuming(observation):
            service._rng.random()
            return predict(observation)

        runtime.predict = consuming
    config = service.cfg
    state = service._rng.bit_generator.state.copy()
    with pytest.raises((ValueError, RuntimeError)):
        policy.infer(observation(policy))
    assert service.cfg is config and service._rng.bit_generator.state == state
    assert "generate_samples_from_batch" not in vars(service.model)
    assert "_get_velocity" not in vars(service.model)
    assert "forward" not in vars(service.model.sampler)
    assert not (policy.output_dir / "request_000000.json").exists()


def test_native_contract_change_and_missing_numeric_route_fail_closed(build):
    policy, runtime = build()
    reset(policy)
    runtime.service.cfg = replace(runtime.service.cfg, guidance=1.0)
    with pytest.raises(ValueError, match="native service"):
        policy.infer(observation(policy))
    with pytest.raises(ValueError, match="not installed"):
        build(
            arm="runtime_selected",
            transform=lambda rt: delattr(rt.service, "_ifl_generation_regions"),
        )


def test_matrix_and_environment_binding(build, monkeypatch, tmp_path):
    policy, _ = build()
    path = tmp_path / "matrix_0.json"
    with pytest.raises(ValueError, match="SHA256"):
        quality.frozen_cell(path, "edge-runtime_default", matrix_sha256="0" * 64)
    data = json.loads(path.read_text())
    data["cells"][0]["expected_runtime_kwargs"]["precision"] = "fp8"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="kwargs"):
        quality.frozen_cell(
            changed, "edge-runtime_default", matrix_sha256=quality._sha(changed)
        )
    monkeypatch.setenv("IFL_UNDECLARED_OPTIMIZER", "1")
    with pytest.raises(ValueError, match="environment"):
        quality.CosmosQualityPolicy.from_matrix(
            path,
            "edge-runtime_default",
            matrix_sha256=quality._sha(path),
            output_dir=tmp_path / "bad-env",
        )
    assert not (tmp_path / "bad-env").exists()


def test_receipt_directory_cannot_be_reopened_and_identity_is_hash_bound(
    build, tmp_path
):
    policy, _ = build()
    assert (
        policy.identity_sha256
        == hashlib.sha256(quality._json_bytes(policy.identity)).hexdigest()
    )
    assert (
        json.loads((policy.output_dir / "identity.json").read_text()) == policy.metadata
    )
    with pytest.raises(FileExistsError):
        quality.CosmosQualityPolicy.from_matrix(
            tmp_path / "matrix_0.json",
            "edge-runtime_default",
            matrix_sha256=quality._sha(tmp_path / "matrix_0.json"),
            output_dir=policy.output_dir,
        )


@pytest.mark.parametrize("seed,bound", [(0, 0), (2**32 - 1, 2), (0, True)])
def test_bad_reset_seed_ranges_rejected_before_public_reset(build, seed, bound):
    policy, runtime = build()
    with pytest.raises(ValueError):
        reset(policy, seed=seed, bound=bound)
    assert runtime.public_resets == [{"prompt": "Initialize Cosmos quality worker"}]


@pytest.mark.parametrize("change", ["disabled", "compile_failed", "unused"])
def test_selected_numeric_fallback_is_not_silently_certified(build, change):
    policy, runtime = build(arm="runtime_selected")
    reset(policy)
    if change == "disabled":
        runtime.service._ifl_conditioning_cache.report()["disabled"] = True
    elif change == "compile_failed":
        runtime.service._ifl_generation_regions.report()["regions"][0]["state"] = (
            "failed"
        )
    else:
        runtime.service._ifl_generation_regions.report()["compiled_calls"] = 0
    with pytest.raises(ValueError, match="Selected"):
        policy.infer(observation(policy))
    assert not (policy.output_dir / "request_000000.json").exists()


def test_changed_implementation_cannot_keep_the_same_identity(build, tmp_path):
    policy, _ = build()
    reset(policy)
    (tmp_path / "native_source.py").write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="source changed"):
        policy.infer(observation(policy))
    assert not (policy.output_dir / "request_000000.json").exists()
    other, _ = build()
    assert other.identity_sha256 != policy.identity_sha256


def test_checkpoint_weights_and_external_load_are_identity_bound(build):
    policy, _ = build()
    record = policy.identity["checkpoint_artifacts"]["model.safetensors"]
    assert record["sha256"] == hashlib.sha256(b"CPU test weight identity").hexdigest()
    assert policy.identity["observed_external_assets"] == [
        {"path": "test-only-Wan.pth", "sha256": "test-observed-asset"}
    ]
    assert policy.identity["source_inventory"]


def test_hub_blob_symlink_origin_is_supported(build, tmp_path):
    policy, runtime = build()
    snapshot = Path(runtime._checkpoint.path)
    target = tmp_path / "blobs" / "sha-configuration"
    target.parent.mkdir()
    target.write_text("{}")
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(target)
    observed = quality._observed_contract(runtime, policy.cell)
    assert observed["checkpoint_config_path"] == str(target)


def test_eager_reference_cannot_carry_runtime_optimizations(build):
    def mutated(runtime):
        runtime.service._ifl_fake_optimization = object()

    with pytest.raises(ValueError, match="must not install"):
        build(arm="eager_native", transform=mutated)
