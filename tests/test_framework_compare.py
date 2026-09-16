"""CPU checks of the fixed foreign-framework contracts and report admission."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from benchmarks.regression import framework_compare as comparison


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks/regression/fixtures/recorded_inputs_v1.npz"


def bootstrap():
    spec = importlib.util.spec_from_file_location("framework_bootstrap_test", ROOT / "scripts/bootstrap_framework_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_read_only_and_does_not_import_model_libraries(tmp_path):
    script = """
import importlib.abc, importlib.util, json, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','numpy','lerobot','vllm','vllm_omni','huggingface_hub'}:
            raise RuntimeError('forbidden: '+fullname)
sys.meta_path.insert(0, Guard())
spec=importlib.util.spec_from_file_location('comparison',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
print(json.dumps(m.plan('vllm-omni-nano')['cell']['effective_schedule']))
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, comparison.__file__], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"action": 4, "guidance": 3.0, "shift": 5.0}
    assert list(tmp_path.iterdir()) == []


def test_safe_fixture_is_exact_historical_decoding_and_feedback():
    old = ROOT / "eval/native_total_2026-09-10/fixtures/va_eval_obs.npz"
    if not old.is_file():
        pytest.skip("Historical conversion evidence is not a public runtime dependency")
    assert comparison.sha(old) == comparison.catalog()["historical_fixture_sha256"]
    current = comparison.Inputs(FIXTURE)
    with np.load(old, allow_pickle=True) as data:
        frames = [data["frame0_0"], *data["jpeg_0"][:12]]
        for i, frame in enumerate(frames):
            for camera, jpeg in enumerate(frame):
                original = np.asarray(Image.open(io.BytesIO(bytes(jpeg))).convert("RGB"))
                assert original.dtype == current.frames[i, camera].dtype
                assert original.tobytes() == current.frames[i, camera].tobytes()
        assert data["actions_0"].tobytes() == current.feedback.tobytes()


def test_all_six_historical_medians_replay_exactly():
    receipts = ROOT / "eval/framework_comparison_2026-09-13/receipts"
    if not receipts.is_dir():
        pytest.skip("Original receipts are audit evidence, not runtime dependencies")
    for cell in comparison.catalog()["cells"]:
        path = receipts / cell["historical"]["receipt_name"]
        assert comparison.sha(path) == cell["historical"]["receipt_sha256"]
        calls = [{**row, "finite": True} for row in json.loads(path.read_text())["calls"]]
        result = comparison.selected_report(cell, calls)
        assert result["p50_ms"] == cell["historical"]["selected_p50_ms"]
        assert result["selected_count"] == (20 if cell["family"] in {"va", "dreamzero"} else 30)
        assert not result["task_quality_certified"]


def test_public_request_protocol_retains_nonobvious_history_and_seed_rules():
    inputs = comparison.Inputs(FIXTURE)
    cells = {row["family"]: row for row in comparison.catalog()["cells"]}
    for i in range(40):
        edge = comparison.request(cells["edge"], inputs, i)
        nano = comparison.request(cells["nano"], inputs, i)
        assert edge["seed"] == nano["seed"] == 1826701615
        assert edge["prompt"] == ("pick up the object" if i < 8 else "place the object down")
        assert np.array_equal(edge["observation"]["observation/joint_position"], np.full(7, .01*(i % 3), np.float32))
    # Original DreamZero reuses frames1..4 at both continuation cycles.
    a = comparison.request(cells["dreamzero"], inputs, 7)
    b = comparison.request(cells["dreamzero"], inputs, 8)
    assert a["prompt"] == b["prompt"] == "pick up the object"
    assert a["observation"]["session_id"] == b["observation"]["session_id"] == "benchmark-2"
    for key in ("observation/exterior_image_0_left", "observation/exterior_image_1_left", "observation/wrist_image_left"):
        assert np.array_equal(a["observation"][key], b["observation"][key])
    va = comparison.request(cells["va"], inputs, 8)
    assert va["observation"]["indices"] == list(range(5, 13))
    assert np.array_equal(va["observation"]["feedback"], inputs.feedback[1])
    assert va["seed"] == 1302 and va["reset"] is False


def test_unqualified_registered_groot_is_not_relabelled_unsupported():
    value = comparison.catalog()
    assert "groot" not in value["unsupported"]["vllm-omni"]
    assert value["not_qualified"]["vllm-omni"]["groot"]["registered_class"] == "Gr00tN1d7Pipeline"
    with pytest.raises(ValueError, match="no measured historical route"):
        comparison.plan("vllm-omni-groot")


@pytest.mark.parametrize("mutation", ["missing_call", "wrong_index", "wrong_phase", "nan", "wrong_shape", "false_finite"])
def test_bad_capture_cannot_produce_selected_latency(mutation):
    cell = comparison.plan("vllm-omni-edge")["cell"]
    calls = [{"i": i, "phase": "warmup" if i < 10 else "measured", "shape": [32, 8], "finite": True, "ms": 100.}
             for i in range(40)]
    if mutation == "missing_call":
        calls.pop()
    elif mutation == "wrong_index":
        calls[-1]["i"] = 0
    elif mutation == "wrong_phase":
        calls[-1]["phase"] = "warmup"
    elif mutation == "nan":
        calls[-1]["ms"] = float("nan")
    elif mutation == "wrong_shape":
        calls[-1]["shape"] = [24, 8]
    else:
        calls[-1]["finite"] = False
    with pytest.raises(ValueError):
        comparison.selected_report(cell, calls)


def synthetic_capture(tmp_path, monkeypatch):
    value = comparison.plan("vllm-omni-edge")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    comparison.write_new(prepared / "preparation.json", {"status": "prepared"})
    comparison.write_new(prepared / "inputs.npz", FIXTURE.read_bytes(), raw=True)
    output = tmp_path / "run"
    output.mkdir()
    assets = {"assets": {"checkpoint": {"path": str(prepared)}}}
    monkeypatch.setattr(comparison, "load_prepared", lambda *a, **k: (value, assets))
    calls, actions = [], {}
    inputs = comparison.Inputs(FIXTURE)
    for i in range(40):
        req = comparison.request(value["cell"], inputs, i)
        action = np.full((32, 8), float(i), np.float32)
        actions[f"action_{i}"] = action
        calls.append({"i": i, "phase": "warmup" if i < 10 else "measured", "shape": [32, 8], "finite": True,
                      "ms": 100. + i, "episode": i, "cycle": 0, "reset": True, "seed": req["seed"],
                      "request_sha256": comparison.request_hash(req),
                      "action_sha256": comparison.hashlib.sha256(action.tobytes()).hexdigest()})
    np.savez(output / "actions.npz", **actions)
    captured = {**value, "status": "captured", "runner_sha256": comparison.sha(comparison.__file__),
                "preparation_sha256": comparison.sha(prepared / "preparation.json"), "calls": calls,
                "actions_sha256": comparison.sha(output / "actions.npz"), "latency": comparison.selected_report(value["cell"], calls),
                "startup_checkpoint_paths": [str(prepared)]}
    comparison.write_new(output / "capture.json", captured)
    comparison.write_new(output / "closed.json", {"status": "closed", "capture_sha256": comparison.sha(output / "capture.json")})
    completion = {"exit_code": 0, "forced_kill": False, "timed_out": False, "runner_sha256": comparison.sha(comparison.__file__),
                  "startup_checkpoint_paths": [str(prepared)]}
    comparison.write_new(output / "completion.json", completion)
    return prepared, output


def test_full_report_reconstructs_actions_requests_and_percentiles(tmp_path, monkeypatch):
    prepared, output = synthetic_capture(tmp_path, monkeypatch)
    result = comparison.report(prepared, output)
    assert result["status"] == "passed"
    assert result["latency"]["p50_ms"] == 124.5
    assert result["task_quality_certified"] is False


@pytest.mark.parametrize("change", ["crash", "timeout", "kill", "no_close", "tampered_actions", "tampered_seed", "wrong_checkpoint_cache"])
def test_completed_capture_without_real_clean_completion_is_rejected(tmp_path, monkeypatch, change):
    prepared, output = synthetic_capture(tmp_path, monkeypatch)
    if change in {"crash", "timeout", "kill"}:
        p = output / "completion.json"
        value = json.loads(p.read_text())
        value.update({"exit_code": -11} if change == "crash" else {"timed_out": True} if change == "timeout" else {"forced_kill": True})
        p.write_bytes(comparison.encoded(value))
    elif change == "no_close":
        (output / "closed.json").unlink()
    elif change == "tampered_actions":
        (output / "actions.npz").write_bytes(b"altered")
    elif change == "wrong_checkpoint_cache":
        p = output / "completion.json"
        value = json.loads(p.read_text())
        value["startup_checkpoint_paths"] = ["/unrelated/cache"]
        p.write_bytes(comparison.encoded(value))
    else:
        p = output / "capture.json"
        value = json.loads(p.read_text())
        value["calls"][-1]["seed"] = 0
        p.write_bytes(comparison.encoded(value))
        (output / "closed.json").write_bytes(comparison.encoded({"status": "closed", "capture_sha256": comparison.sha(p)}))
    with pytest.raises((ValueError, FileNotFoundError)):
        comparison.report(prepared, output)


def test_exact_startup_patch_refuses_source_drift_and_is_not_implicit(tmp_path):
    module = bootstrap()
    upstream = Path(os.environ.get(
        "VLLM_OMNI_ROOT", "/home/ubuntu/work_clones/vllm-omni-benchmark-20260913"))
    if not upstream.is_dir():
        pytest.skip("Independent upstream source audit fixture not present")
    patch = json.loads((comparison.DATA / comparison.catalog()["startup_patch_file"]).read_text())
    for item in patch["changes"]:
        path = tmp_path / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((upstream / item["path"]).read_bytes())
    with pytest.raises(ValueError, match="requires"):
        module.apply_startup_patch(tmp_path, acknowledged=False)
    module.apply_startup_patch(tmp_path, acknowledged=True)
    for item in patch["changes"]:
        assert comparison.sha(tmp_path / item["path"]) == item["after_sha256"]
    with pytest.raises(ValueError, match="unexpected original"):
        module.apply_startup_patch(tmp_path, acknowledged=True)


def test_bootstrap_plan_has_public_pins_and_no_author_paths(tmp_path):
    module = bootstrap()
    result = module.make_plan("vllm-omni", tmp_path / "new", "/usr/bin/python3.12", "uv", allow_startup_patch=True)
    assert not (tmp_path / "new").exists()
    assert result["sources"]["vllm-omni"]["revision"] == "f7d9deb45ab56e6a2ccc1690279bd9e6bdefbfe3"
    assert result["sources"]["cosmos-framework"]["revision"] == "2b6c9a7061ae78dc83e29a4910ec5f8c9fe4b6ce"
    assert any("vllm @ https://files.pythonhosted.org/" in p and "#sha256=" in p for p in result["public_requirements"])
    for name in ("catalog.json", "sources.json", "omni_thor_startup_patch.json", "omni_thor_startup_patch_portable_v2.json"):
        raw = (comparison.DATA / name).read_text()
        assert "/home/guanming" not in raw and "/home/ubuntu" not in raw
    assert not result["gpu_qualified"]
    assert "iopath==0.1.10" in result["constraints"]
    assert "portalocker==4.3.0" in result["constraints"]
    assert any("iopath @ https://files.pythonhosted.org/" in p and
               "#sha256=3311c16a4d9137223e20f141655759933e1eda24f8bff166af834af3c645ef01" in p
               for p in result["public_requirements"])


def test_cosmos_metadata_only_adds_required_inference_utility(tmp_path):
    module = bootstrap()
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\ndependencies = [\n  "transformers>=4.57.1,<5.0.0",\n]\n')
    native = tmp_path / "model.py"
    native.write_bytes(b"untouched native implementation\n")
    receipt = module.cosmos_metadata(tmp_path)
    assert path.read_text() == ('[project]\ndependencies = [\n  "transformers>=5.13.0,<5.15",\n'
                                '    "iopath==0.1.10",\n]\n')
    assert receipt["after_sha256"] == module.sha(path)
    assert native.read_bytes() == b"untouched native implementation\n"
    with pytest.raises(ValueError, match="unexpected Cosmos"):
        module.cosmos_metadata(tmp_path)


@pytest.mark.parametrize("operation,expected", [
    ("pass", None),
    ("__import__('ctypes.util', fromlist=['find_library']).find_library('dl')", None),
    ("torch.cuda.init()", "torch.cuda.initialization"),
    ("open('missing-model.pth', 'rb')", "open"),
    ("__import__('socket').getaddrinfo('example.com', 443)", "socket.getaddrinfo"),
    ("__import__('subprocess').run(['uname', '-p'])", "subprocess.Popen"),
])
def test_native_utility_cpu_guard_rejects_caught_side_effects(tmp_path, operation, expected):
    # Use the actual public guard in a fresh interpreter: catching a denied
    # operation inside a dependency must not turn its check into a pass.
    package = tmp_path / "cosmos_framework/data/generator/action/utils"
    package.mkdir(parents=True)
    for parent in [package, *package.parents]:
        if parent == tmp_path:
            break
        (parent / "__init__.py").write_text("")
    (package / "transforms.py").write_text(
        "import torch\ntry:\n    " + operation + "\nexcept Exception:\n    pass\n"
        "class ActionTransformPipeline: pass\n")
    (tmp_path / "torch.py").write_text('''
from types import SimpleNamespace
def stub(*args, **kwargs): pass
cuda = SimpleNamespace(_lazy_init=stub, init=stub, set_device=stub, current_device=stub,
    get_device_capability=stub, get_device_properties=stub, is_initialized=lambda: False)
''')
    receipt = tmp_path / "receipt.json"
    code = '''
import importlib.util, sys
sys.path.insert(0, sys.argv[1])
spec=importlib.util.spec_from_file_location('tested',sys.argv[2]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.verify_cosmos_transform(sys.argv[3])
'''
    process = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path),
                              str(ROOT / "scripts/bootstrap_framework_compare.py"), str(receipt)],
                             capture_output=True, text=True)
    actual = json.loads(receipt.read_text())
    assert not actual["gpu_inference"]
    assert not actual["model_constructed"]
    assert actual["status"] == ("passed" if expected is None else "failed")
    assert process.returncode == (0 if expected is None else 1)
    assert actual["blocked_operations"] == ([] if expected is None else [expected])


def test_clean_environment_does_not_inherit_selected_flash_or_omni_overrides():
    result = comparison.child_environment({"IFL_COSMOS_DYNAMIC_STEP_CACHE": "1", "VLLM_ATTENTION_BACKEND": "other",
                                           "PYTHONPATH": "/private", "TRITON_PTXAS_PATH": "/old", "CUDA_VISIBLE_DEVICES": "0",
                                           "PATH": "/usr/bin"})
    assert result["CUDA_VISIBLE_DEVICES"] == "0"
    assert not {"IFL_COSMOS_DYNAMIC_STEP_CACHE", "VLLM_ATTENTION_BACKEND", "PYTHONPATH", "TRITON_PTXAS_PATH"} & result.keys()


def test_installed_source_guard_rejects_native_drift_before_import(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    package = tmp_path / "env" / "lib" / "lerobot"
    package.mkdir(parents=True)
    source = package / "model.py"
    source.write_text("raise RuntimeError('must not be imported')\n")
    facts = {"lerobot": {"package": "lerobot", "revision": "a" * 40,
                         "installed_inventory": {"model.py": comparison.sha(source)}, "runtime_patch": None}}
    (data / "sources.json").write_bytes(comparison.encoded(facts))
    monkeypatch.setattr(comparison, "DATA", data)
    monkeypatch.setattr(comparison.sys, "prefix", str(tmp_path / "env"))
    monkeypatch.setattr(comparison.importlib.util, "find_spec", lambda name: type("Spec", (), {"submodule_search_locations": [str(package)]}))
    assert comparison.installed_sources("lerobot")["lerobot"]["revision"] == "a" * 40
    source.write_text("changed arithmetic\n")
    with pytest.raises(ValueError, match="source mismatch"):
        comparison.installed_sources("lerobot")


def test_va_shape_cannot_be_swapped_between_initial_and_continuation():
    cell = comparison.plan("lerobot-va")["cell"]
    calls = [{"i": i, "phase": "warmup" if i < 3 else "measured", "shape": [1, 16 if i % 3 == 0 else 32, 16],
              "finite": True, "ms": 100.} for i in range(33)]
    bad = deepcopy(calls)
    bad[4]["shape"] = [1, 16, 16]
    comparison.selected_report(cell, calls)
    with pytest.raises(ValueError, match="return layout"):
        comparison.selected_report(cell, bad)


def test_auxiliary_cache_is_new_pinned_and_keeps_existing_refs_untouched(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    data = original / "tokenizer.json"
    data.write_text('{"tokenizer": "fixed"}')
    ref = original / "main"
    ref.write_text("unrelated existing revision")
    spec = {"bytes": data.stat().st_size, "sha256": comparison.sha(data)}
    cell = {"auxiliary_repositories": {"public/tokenizer": {"revision": "a" * 40, "files": {"tokenizer.json": spec}}}}
    requests = []

    def resolve(**kwargs):
        requests.append(kwargs)
        return str(data)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=resolve))
    destination = tmp_path / "prepared"
    destination.mkdir()
    result = comparison.prepare_auxiliary(cell, destination, cache_dir=str(original), local_files_only=True)
    assert requests == [{"repo_id": "public/tokenizer", "filename": "tokenizer.json", "revision": "a" * 40,
                         "cache_dir": str(original), "local_files_only": True}]
    cache = Path(result["cache"])
    assert (cache / "models--public--tokenizer/refs/main").read_text() == "a" * 40
    assert comparison.sha(Path(result["files"][0]["path"])) == spec["sha256"]
    assert ref.read_text() == "unrelated existing revision"
    with pytest.raises(FileExistsError):
        comparison.prepare_auxiliary(cell, destination, cache_dir=str(original), local_files_only=True)


def test_umt5_preparation_is_tokenizer_only_and_pi05_aux_is_pinned():
    dreamzero = comparison.plan("vllm-omni-dreamzero")["cell"]
    assert set(dreamzero["assets"]["tokenizer"]["files"]) == {
        "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"}
    assert all(not name.endswith((".safetensors", ".bin")) for name in dreamzero["assets"]["tokenizer"]["files"])
    pi05 = comparison.plan("lerobot-pi05")["cell"]
    assert pi05["auxiliary_repositories"]["google/paligemma-3b-pt-224"]["revision"] == "35e4f46485b4d07967e7e9935bc3786aad50687c"


def test_bootstrap_selects_compatible_build_metadata_and_fresh_tmpfs_env(tmp_path):
    m = bootstrap()
    result = m.make_plan("lerobot", tmp_path / "evidence", "/python", "uv", env_dir=tmp_path / "ram-env")
    assert result["environment"] == str(tmp_path / "ram-env")
    assert "setuptools-scm==8.3.1" in result["build_requirements"]
    assert result["wheel_metadata_repair"] == "nvidia-cusparselt-cu13-0.8.0-aarch64"
    with pytest.raises(ValueError, match="must not contain"):
        m.make_plan("lerobot", tmp_path / "evidence", "/python", "uv", env_dir=tmp_path)


def test_framework_specific_native_repair_cannot_be_swapped(tmp_path):
    m = bootstrap()
    omni = m.make_plan("vllm-omni", tmp_path / "omni", "/python", "uv", allow_startup_patch=True)
    assert omni["wheel_metadata_repair"] == "nvidia-cusparselt-cu13-0.8.1-aarch64"
    profiles, _ = m.load()
    repairs = profiles["native_wheel_repairs"]
    assert repairs["lerobot"]["rule"]["dist_info"] == "nvidia_cusparselt_cu13-0.8.0.dist-info"
    assert repairs["vllm-omni"]["rule"]["dist_info"] == "nvidia_cusparselt_cu13-0.8.1.dist-info"
    original = tmp_path / "already-audited-wheel.whl"
    original.write_bytes(b"preserved public artifact placeholder")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(comparison.encoded({
        "status": "passed", "all_other_payload_bytes_preserved": True,
        "repair_source_sha256": m.sha(m.REPO / "scripts/repair_vendor_wheel.py"),
        "public_source": repairs["lerobot"]["rule"],
        "repaired": {"sha256": repairs["lerobot"]["repaired_sha256"]}}))
    root = tmp_path / "new-evidence"
    root.mkdir()
    with pytest.raises(ValueError, match="not the audited public artifact"):
        m.repaired_cusparselt(root, "vllm-omni", wheel=original, receipt_path=receipt)
    assert original.read_bytes() == b"preserved public artifact placeholder"
    assert not (root / "wheel_repair/receipt.json").exists()


def portable_memory_query():
    patch = json.loads((comparison.DATA / comparison.catalog()["startup_patch_file"]).read_text())
    replacement = patch["changes"][1]["new"]
    namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (12345, 99999)))}
    # Execute the exact published replacement body with a CPU fake memory query.
    exec("def query(self):\n    " + replacement.replace("\n        ", "\n    "), namespace)
    return namespace["query"]


def sparse_checkpoint(root):
    root.mkdir(parents=True)
    big = root / "model.safetensors"
    with big.open("wb") as stream:
        stream.write(b"unchanged")
        stream.truncate(64 * 1024 * 1024)
    (root / "tiny.json").write_text("{}")
    return big


def test_portable_startup_advice_binds_actual_checkpoint_not_processor_cache(tmp_path, monkeypatch):
    checkpoint = tmp_path / "arbitrary-model-directory"
    weight = sparse_checkpoint(checkpoint)
    (checkpoint / "second-alias").symlink_to(weight)
    processor = tmp_path / "separate-processor-cache"
    sparse_checkpoint(processor)
    monkeypatch.setenv("HF_HUB_CACHE", str(processor))
    prepared = {"assets": {"checkpoint": {"path": str(checkpoint)}}, "auxiliary": {"cache": str(processor)}}
    paths = comparison.checkpoint_cache_paths({"framework": "vllm-omni"}, prepared)
    monkeypatch.setenv("VLLM_OMNI_CHECKPOINT_PATHS", json.dumps(paths))
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, offset, count, advice: calls.append(
        (Path(os.readlink(f"/proc/self/fd/{fd}")), offset, count, advice)))
    assert portable_memory_query()(SimpleNamespace(device="cpu-fixture")) == 12345
    assert calls == [(weight, 0, 0, os.POSIX_FADV_DONTNEED)]
    assert weight.open("rb").read(9) == b"unchanged"
    monkeypatch.setenv("VLLM_OMNI_CHECKPOINT_PATHS", '["relative-unbound-path"]')
    with pytest.raises(ValueError, match="existing absolute"):
        portable_memory_query()(SimpleNamespace(device="cpu-fixture"))
    assert len(calls) == 1


@pytest.mark.parametrize("selection", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "default"])
def test_portable_startup_fallback_respects_selected_public_cache(tmp_path, monkeypatch, selection):
    for key in ("VLLM_OMNI_CHECKPOINT_PATHS", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    cache = tmp_path / "custom-cache" if selection != "default" else tmp_path / "home/.cache/huggingface/hub"
    weight = sparse_checkpoint(cache / "models--public--checkpoint/blobs")
    if selection != "default":
        monkeypatch.setenv(selection, str(cache))
    calls = []
    monkeypatch.setattr(os, "posix_fadvise", lambda fd, *_: calls.append(Path(os.readlink(f"/proc/self/fd/{fd}"))))
    assert portable_memory_query()(SimpleNamespace(device="cpu-fixture")) == 12345
    assert calls == [weight]
