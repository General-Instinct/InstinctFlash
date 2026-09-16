"""CPU-only validation of immutable FP32 scales and GEMM state envelopes."""
import copy
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_state", ROOT / "serving/flash_rt/core/frozen_state.py")
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def payload():
    return {"schema": state.SCHEMA, "identity": {"weights": "model", "gpu": "5090"},
            "prompt": "move the bowl", "calibration": {"observations_sha256": ["a" * 64], "percentile": 99.0},
            "scale_bits": state.scale_bits({"layer": .125}), "token_lengths": [150, 151],
            "gemm": {"identity": "test", "entries": [[0, 16, 32, 64, "00" * 64]]}}


def test_state_round_trip_is_exact_and_refuses_overwrite(tmp_path):
    value = payload()
    path = tmp_path / "state.json"
    checksum = state.save(path, value)
    assert checksum == state.digest(value)
    assert state.load(path, value["identity"]) == value
    assert state.decode_scales(value["scale_bits"]) == {"layer": .125}
    with pytest.raises(FileExistsError):
        state.save(path, value)
    assert len(list(tmp_path.iterdir())) == 1


def test_checksum_and_environment_are_checked(tmp_path):
    value = payload()
    document = state.seal(value)
    changed = copy.deepcopy(document)
    changed["payload"]["prompt"] = "another prompt"
    with pytest.raises(ValueError, match="checksum"):
        state.validate(changed)
    with pytest.raises(ValueError, match="identity"):
        state.validate(document, {"weights": "different"})


@pytest.mark.parametrize("mutation", ["schema", "extra", "scale_nan", "scale_zero", "scale_size", "scale_hex",
    "profiles_empty", "profiles_duplicate", "profiles_bool", "profiles_outside", "gemm_duplicate",
    "gemm_type", "gemm_dimension", "gemm_blob", "gemm_empty", "provenance"])
def test_resealed_invalid_state_is_still_rejected(mutation):
    value = payload()
    if mutation == "schema": value["schema"] = "unknown"
    elif mutation == "extra": value["unexpected"] = 1
    elif mutation == "scale_nan": value["scale_bits"]["layer"] = "0000c07f"
    elif mutation == "scale_zero": value["scale_bits"]["layer"] = "00000000"
    elif mutation == "scale_size": value["scale_bits"]["layer"] = "00"
    elif mutation == "scale_hex": value["scale_bits"]["layer"] = "zzzzzzzz"
    elif mutation == "profiles_empty": value["token_lengths"] = []
    elif mutation == "profiles_duplicate": value["token_lengths"] = [150, 150]
    elif mutation == "profiles_bool": value["token_lengths"] = [True]
    elif mutation == "profiles_outside": value["token_lengths"] = [201]
    elif mutation == "gemm_duplicate": value["gemm"]["entries"] *= 2
    elif mutation == "gemm_type": value["gemm"]["entries"][0][0] = 999
    elif mutation == "gemm_dimension": value["gemm"]["entries"][0][1] = 0
    elif mutation == "gemm_blob": value["gemm"]["entries"][0][4] = "ff"
    elif mutation == "gemm_empty": value["gemm"]["entries"] = []
    else: value["calibration"] = {}
    with pytest.raises(ValueError):
        state.validate(state.seal(value))


@pytest.mark.parametrize("mutation", [None, "missing_active", "extra_unused", "unknown"])
def test_frontend_restore_checks_only_the_250_executed_scale_names(tmp_path, mutation):
    # Execute the real restore method without importing Torch/model dependencies.
    source = ast.parse((ROOT / "serving/flash_rt/frontends/torch/pi05_frozen.py").read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_state")
    namespace = {"state": state, "ENC_L": 18}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "real_load_state", "exec"), namespace)
    weights = {f"vision_{kind}_w_{i}" for kind in ("attn_qkv", "attn_o", "ffn_up", "ffn_down") for i in range(27)}
    weights |= {f"{part}_{kind}_w_{i}" for part in ("encoder", "decoder")
                for kind in ("attn_qkv", "attn_o", "ffn_gate_up", "ffn_down") for i in range(18)}
    weights.add("vision_projector_w")
    unused = {"encoder_attn_o_w_17", "encoder_ffn_gate_up_w_17", "encoder_ffn_down_w_17"}
    scales = {k: .125 for k in weights - unused}
    assert len(weights) == 253 and len(scales) == 250
    if mutation == "missing_active": scales.pop("encoder_attn_o_w_16")
    elif mutation == "extra_unused": scales["encoder_attn_o_w_17"] = .125
    elif mutation == "unknown": scales["not_a_layer"] = .125
    value = payload()
    value["scale_bits"] = state.scale_bits(scales)
    path = tmp_path / "state.json"
    state.save(path, value)
    calls = []
    gemm = SimpleNamespace(export_algo_cache=lambda: [], algo_cache_identity=lambda: "test",
                           import_algo_cache=lambda *args: calls.append(args))
    model = SimpleNamespace(_profiles={}, _scales=None, identity=lambda: value["identity"],
        frontend=SimpleNamespace(gemm=gemm, _fp8_weights=dict.fromkeys(weights)),
        set_prompt=lambda p: None)
    if mutation:
        with pytest.raises(ValueError, match="layer set"):
            namespace["load_state"](model, path)
        assert not calls
    else:
        namespace["load_state"](model, path)
        assert model._scales == scales and len(calls) == 1


def test_frozen_rollout_comparison_refuses_duplicate_seeds_and_detects_action_drift():
    from benchmarks.vla.pi05_frozen_replay import compare_rollouts
    rows = [{"seed": seed, "success": True, "steps": 10,
             "scene": {"seed": seed, "init_state_index": seed % 50},
             "noise_sha256": ["a" * 64], "action_digest": "b" * 64}
            for seed in range(40100, 40150)]
    assert compare_rollouts(rows, rows)["status"] == "PASS"
    changed = copy.deepcopy(rows)
    changed[0]["action_digest"] = "c" * 64
    assert compare_rollouts(rows, changed)["mismatch_seeds"] == [40100]
    changed = copy.deepcopy(rows)
    changed[-1] = changed[0]
    with pytest.raises(ValueError, match="unique"):
        compare_rollouts(rows, changed)
