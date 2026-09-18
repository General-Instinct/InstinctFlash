"""CPU seed contracts through the real backend, adapter, and DROID builder.

Only CUDA discovery, model residency, and the vendor model service are stand-ins.
The random stream uses real NumPy; these tests do not measure model actions.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "cosmos3_policy"))

from cosmos3_iwm.adapter import Cosmos3PolicyAdapter, MODEL_ID, REQUIRED_SERVING_KEYS
from instinctflash.descriptors.known import lookup
from instinctflash.runtime.execution import InProcessBackend


@pytest.fixture
def build_backend(monkeypatch, tmp_path):
    # Ignore optional optimizations enabled by the developer's shell.
    for name in os.environ.copy():
        if name.startswith("IFL_COSMOS3_") or name == "IFL_BF16_LINEAR_RELU2":
            monkeypatch.delenv(name)
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True,
                                 get_device_capability=lambda: (9, 0),
                                 manual_seed_all=lambda seed: None)
    torch.manual_seed = lambda seed: None
    monkeypatch.setitem(sys.modules, "torch", torch)

    nano = ModuleType("cosmos3_iwm.nano_action_only")
    nano.nano_action_only_construction = lambda **kwargs: nullcontext()
    nano.verify_nano_action_only_model = lambda model: 0
    residency = ModuleType("cosmos3_iwm.sm89_residency")
    residency.cpu_construction = lambda **kwargs: nullcontext()
    residency.finalize_service = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, nano.__name__, nano)
    monkeypatch.setitem(sys.modules, residency.__name__, residency)

    services = []

    class VendorService:
        def __init__(self, cfg):
            self.cfg = cfg
            self._lock = threading.Lock()
            self._rng = np.random.default_rng(cfg.seed)
            services.append(self)

        def infer(self, observation):
            return {"action": self._rng.random((32, 8))}

    vendor = ModuleType("cosmos_framework.scripts.action_policy_server_robolab")
    vendor.RobolabServerArgs = SimpleNamespace
    vendor.RobolabPolicyService = VendorService
    monkeypatch.setitem(sys.modules, vendor.__name__, vendor)

    # Metadata-only fixtures satisfy path resolution; no weights are loaded.
    for name in ("config.json", "model.safetensors.index.json", "checkpoint.json"):
        (tmp_path / name).write_text(json.dumps({}))
    declared = lookup(MODEL_ID)["execution"]
    backends = []

    def build(runtime_seed, checkpoint_seed=7):
        extra = {key: declared[key] for key in REQUIRED_SERVING_KEYS}
        if checkpoint_seed is not None:
            extra["seed"] = checkpoint_seed
        checkpoint = SimpleNamespace(
            path=tmp_path, model_id=MODEL_ID,
            execution=SimpleNamespace(extra=extra, model_id=MODEL_ID,
                                      backbone="cosmos3_policy"))
        backend = InProcessBackend(Cosmos3PolicyAdapter(), checkpoint,
                                   SimpleNamespace(results=[]),
                                   device="cuda", seed=runtime_seed)
        backends.append(backend)
        backend.reset(prompt="pick up the cup")
        return backend, services[-1]

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield build
    finally:
        for backend in backends:
            backend.close()
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@pytest.mark.parametrize("seed", [0, 11, 22])
def test_runtime_seed_overrides_checkpoint_seed(build_backend, seed):
    _, service = build_backend(seed, checkpoint_seed=7)
    assert service.cfg.seed == seed


@pytest.mark.parametrize("checkpoint_seed, expected", [(29, 29), (None, 0)])
def test_unspecified_runtime_seed_preserves_checkpoint_default(
        build_backend, checkpoint_seed, expected):
    _, service = build_backend(None, checkpoint_seed)
    assert service.cfg.seed == expected


@pytest.mark.parametrize("runtime_seed, expected", [(11, 11), (None, 7)])
def test_seed_stream_advances_and_episode_reset_replays(
        build_backend, runtime_seed, expected):
    backend, _ = build_backend(runtime_seed, checkpoint_seed=7)
    stream = np.random.default_rng(expected)
    first = backend.predict({})["action"]
    second = backend.predict({})["action"]
    np.testing.assert_array_equal(first, stream.random((32, 8)))
    np.testing.assert_array_equal(second, stream.random((32, 8)))
    assert not np.array_equal(first, second)
    backend.reset(prompt="pick up the cup")
    np.testing.assert_array_equal(first, backend.predict({})["action"])


def test_explicit_seed_is_independent_of_checkpoint_seed(build_backend):
    first, _ = build_backend(11, checkpoint_seed=7)
    second, _ = build_backend(11, checkpoint_seed=29)
    np.testing.assert_array_equal(first.predict({})["action"],
                                  second.predict({})["action"])
