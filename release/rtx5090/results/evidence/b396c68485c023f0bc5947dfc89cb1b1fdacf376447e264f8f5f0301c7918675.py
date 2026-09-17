"""Independent, CPU-only audit of one actual RTX5090 edge off-box archive.

Reads bytes and NumPy arrays only. Never imports model code, probes a device,
connects to the host, extracts archive paths or launches a capture.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

import capture_audit
from capture_audit import digest, require
from producer_contracts import validate_config, validate_primitive
from serving_audit import archive_files, serving

HERE = Path(__file__).resolve().parent
TARGET = {"name": "rtx5090", "capability": [12, 0]}
GPU = "GPU-004614f4-173a-187b-4e67-a2e1799e4d42"
SOURCE = "0a53125c5af6d76260be96e39077cdcc3c01126d"
WHEEL_SOURCE = "2f38d095fc2a222a7b51ace7219e4cbeed4d7123"
WORKER_SHA = "fbd9c063cbcbb49aca1304aa256ab8a85441a39e6fa99afb78c5b94bdfbf2709"
RUN = "/workspace/ifl_rtx5090_20260916/edge-model-e2e-v1"
PREFIX = "edge-model-e2e-v1/"
CONTINUITY_SHA = "4032e4e28ba3c28aed038b30c5c1dd20b566deee2848ba5a952afd3ffae8b5ae"


def load_local(name):
    return json.loads((HERE / name).read_bytes())


def check_hash(raw, expected, label):
    require(isinstance(expected, str) and len(expected) == 64 and digest(raw) == expected,
            "Byte hash differs: " + label)


def validate_bound_config(config):
    validate_config(config)
    expected = load_local("edge.template.json")
    for key in ("bootstrap_completion", "primitive_report", "input_completion"):
        expected[key]["sha256"] = config[key]["sha256"]
    require(config == expected, "Actual config differs from frozen template beyond the three actual pending receipt SHAs")
    require(config["source_commit"] == SOURCE and config["output"] == RUN, "Wrong actual source/run")


def hardware(row):
    require(row["status"] == "passed" and row["target"] == TARGET
            and row["capability"] == [12, 0] and row["name"] == "NVIDIA GeForce RTX 5090"
            and row["uuid"].removeprefix("GPU-") == GPU.removeprefix("GPU-"),
            "Wrong actual SM120 GPU identity")


def request_hash(value):
    # Exact public input wire hashing, copied as a pure byte contract.
    h = hashlib.sha256()
    def add(item):
        if isinstance(item, dict):
            h.update(b"dict:")
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(f"list:{len(item)}:".encode())
            for child in item:
                add(child)
        elif isinstance(item, np.ndarray):
            h.update(json.dumps(["array", item.dtype.str, item.shape]).encode())
            h.update(np.ascontiguousarray(item).tobytes())
        else:
            h.update(json.dumps([type(item).__name__, item], sort_keys=True).encode())
    add(value)
    return h.hexdigest()


def prompt(episode):
    return "pick up the object" if episode % 2 == 0 else "place the object down"


def observation(frames, i):
    from PIL import Image
    return {"image": np.asarray(Image.fromarray(frames[1 + i % 12][0]).resize((640, 540))),
            "state": np.full(8, .01 * (i % 3), np.float32), "prompt": prompt(i)}


def validate_requests(receipt, frames):
    require("queue_drain" not in receipt, "EDGE must retain native complete action blocks, not pi05 queue metadata")
    require(len(receipt["cases"]) == len(receipt["calls"]) == 25, "Missing EDGE predictions")
    for i, (case, call) in enumerate(zip(receipt["cases"], receipt["calls"])):
        expected = {"i": i, "episode": i, "cycle": 0, "phase": "warmup" if i < 5 else "measured",
            "seed": 707 + i, "input_sha256": request_hash({"observation": observation(frames, i), "prompt": prompt(i)}),
            "feedback_sha256": request_hash({}), "call_kind": "generation"}
        require(case == expected, "Fixed EDGE camera/state/RNG/reset signature differs")
        require(call["i"] == i and call["shape"] == [32, 8], "Native full 32x8 action horizon differs")
        require(np.isfinite(call["ms"]) and call["ms"] > 0, "Invalid EDGE call duration")


def installed_sources(installed, config, stage, bootstrap, provenance):
    expected = load_local("expected_source_inventory.json")
    require(installed["passed"] is True and installed["GPU_used"] is False
            and installed["model_imported"] is False and installed["python"] == config["python"]
            and installed["dependency_distributions"] == config["expected_distributions"], "Actual installed probe differs")
    prefix = str(Path(config["python"]).parent.parent)
    require(installed["prefix"] == prefix and installed["python_version"].startswith("3.13."), "Wrong model interpreter")
    require(set(installed["distributions"]) == set(expected["public_distributions"]), "Public installed inventory incomplete")
    path_hashes = {}
    for name, frozen in expected["public_distributions"].items():
        actual = installed["distributions"][name]
        require(actual["version"] == frozen["version"] and actual["wheel_sha256"] == frozen["wheel_sha256"],
                "Installed public wheel identity differs")
        require(set(actual["files"]) == set(frozen["files"]), "Installed wheel member closure differs")
        stage_rows = [row for row in stage["wheels"] if row["inspection"]["distribution"].replace("_", "-").lower() == name]
        require(len(stage_rows) == 1 and stage_rows[0]["sha256"] == frozen["wheel_sha256"], "Stage public wheel differs")
        require(any(row["sha256"] == frozen["wheel_sha256"] for row in bootstrap["wheels"]), "Bootstrap did not install bound public wheel")
        for member, info in frozen["files"].items():
            row = actual["files"][member]
            require(row["sha256"] == info["sha256"] and row["staged_relative"] == info["staged_relative"]
                    and stage["files"][info["staged_relative"]]["sha256"] == info["sha256"]
                    and row["path"] == prefix + "/lib/python3.13/site-packages/" + member,
                    "Installed/source/wheel member binding differs: " + member)
            path_hashes[row["path"]] = info["sha256"]
    require(len(path_hashes) == 432, "Expected complete 432-member edge source closure")
    require(provenance["passed"] is True and provenance["family"] == "edge"
            and provenance["source_commit"] == SOURCE and provenance["vendor_revision"] == "f734253f0f6af3e268372402f44435c38f55ef3e"
            and provenance["bootstrap_sha256"] == config["bootstrap_completion"]["sha256"], "Wrong actual vendor CPU provenance")
    require(provenance["GPU_used"] is False and provenance["model_imported"] is False
            and provenance["source_patch_verified"] is True and provenance["packaging_patch_verified"] is True,
            "Incomplete original Git/patch/packaging proof")
    require(any(row["sha256"] == provenance["wheel_sha256"] for row in bootstrap["wheels"]), "Bootstrap vendor wheel differs")
    for path, value in provenance["expected_path_sha256"].items():
        if path.endswith(".py"):
            require(expected["vendor_paths"].get(path) == value, "Vendor installed/source Python bytes differ")
    path_hashes.update(provenance["expected_path_sha256"])
    return path_hashes


def validate_input_rows(inputs, assets, config):
    frozen = load_local("original_input_manifest.json")
    require(inputs["status"] == "complete" and inputs["target"] == "rtx5090" and inputs["family"] == "edge"
            and inputs["cache_dir"] == config["input_cache"] and inputs["source_byte_inventory_unchanged"] is True
            and inputs["source_manifest_sha256"] == config["input_source_manifest_sha256"]
            and inputs["file_count"] == 35 and inputs["total_asset_bytes"] == 11990822717,
            "Exact original input completion differs")
    keys = ("repository", "revision", "filename", "bytes", "sha256")
    normalized = lambda rows: sorted(tuple(row[key] for key in keys) for row in rows)
    require(normalized(inputs["files"]) == normalized(frozen["files"]), "Original 35-file inventory changed")
    require(assets["status"] == "assets_hash_verified" and assets["offline"] is True
            and assets["model_constructed"] is False, "Public full-hash asset preparation incomplete")
    primary = assets["primary_checkpoint"]
    rows = [dict(row, repository=primary["model_id"], revision=primary["revision"]) for row in primary["files"]]
    rows += assets["files"]
    require(normalized(rows) == normalized(frozen["files"]), "Actual public asset byte identity differs")
    require(all(row["materialization"] == "hardlink" for row in rows), "Asset placement differs from bound worker")
    return {"files": 35, "bytes": 11990822717, "full_file_sha_receipts_verified": True,
            "post_load_tensor_value_comparison": "not part of the existing edge contract",
            "scope": "Exact original byte receipts plus pinned native-loader and installed source; no new weight read or tensor load."}


def audit(path, binding, launch, provenance):
    require(launch["schema"] == "instinctflash.rtx5090_edge_audit_launch_binding.v1"
            and launch["source_commit"] == SOURCE and launch["target"] == TARGET and launch["GPU_uuid"] == GPU
            and launch["worker_sha256"] == WORKER_SHA
            and launch["template_sha256"] == "89d40032ae42629b0ad6287fe69a7724ebb1c6c9ea66fbabe0d03f1d3a6d8b62", "Missing/wrong actual EDGE launch binding")
    require(type(launch["worker_pid"]) is int and launch["worker_pid"] > 0
            and isinstance(launch["worker_start_ticks"], str) and launch["worker_start_ticks"].isdigit()
            and all(isinstance(launch[key], str) and len(launch[key]) == 64
                    and set(launch[key]) <= set("0123456789abcdef")
                    for key in ("config_sha256", "input_completion_sha256", "vendor_provenance_sha256", "bootstrap_completion_sha256", "primitive_report_sha256")), "Actual EDGE launch fields remain pending")
    require(binding["schema"] == "instinctflash.rtx5090_edge_archive_binding.v1", "Wrong actual archive binding schema")
    require(binding["target"] == TARGET and binding["source_commit"] == SOURCE and binding["gpu_uuid"] == GPU,
            "Wrong actual archive target/source/device binding")
    files, manifest, archive_sha = archive_files(path)
    check_hash(Path(path).read_bytes(), binding["archive_sha256"], "whole off-box archive")
    check_hash(files["provenance/archive_manifest.json"], binding["archive_manifest_sha256"], "archive manifest")
    require(manifest["source_run"] == RUN and manifest["source_commit"] == SOURCE, "Wrong archived source run")
    raw = lambda name: files[PREFIX + name]
    read = lambda name: json.loads(raw(name))
    config, completion = read("config.json"), read("completion.json")
    validate_bound_config(config)
    check_hash(raw("config.json"), launch["config_sha256"], "actual launched EDGE config")
    require(config["input_completion"]["sha256"] == launch["input_completion_sha256"], "Actual launched input receipt differs")
    require(config["bootstrap_completion"]["sha256"] == launch["bootstrap_completion_sha256"]
            and config["primitive_report"]["sha256"] == launch["primitive_report_sha256"], "Actual launch bootstrap/primitive differs")
    check_hash(raw("config.json"), binding["config_sha256"], "actual bound config")
    check_hash(raw("completion.json"), binding["completion_sha256"], "actual terminal completion")
    require(binding["completion_sha256"] == manifest["source_completion_sha256"], "Manifest completion differs")
    require(completion["schema"] == "instinctflash.rtx5090_model_e2e.v1" and completion["family"] == "edge"
            and completion["status"] == "complete" and completion["passed"] is True
            and completion["lease_released"] is True and completion["owned_processes_terminal"] is True
            and completion["automatic_retries"] == 0 and completion["source_commit"] == SOURCE
            and completion["target"] == TARGET and completion["GPU_uuid"] == GPU
            and completion["worker_sha256"] == WORKER_SHA, "Outer worker did not pass/release its children and leases")
    check_hash(raw("config.json"), completion["config_sha256"], "worker config")
    identity = binding["worker"]
    require(identity["pid"] == launch["worker_pid"] and identity["start_ticks"] == launch["worker_start_ticks"], "Different EDGE launch identity")
    require(type(identity["pid"]) is int and identity["pid"] > 0
            and isinstance(identity["start_ticks"], str) and identity["start_ticks"].isdigit()
            and identity["exited"] is True and completion["worker_pid"] == identity["pid"]
            and completion["worker_start_ticks"] == identity["start_ticks"], "Actual launch/terminal process identity differs")
    check_hash(files["provenance/worker.py"], WORKER_SHA, "actual private worker")
    check_hash(files["provenance/installed_source_probe.py"], config["installed_probe_sha256"], "installed source probe")
    inputs = {}
    for key, name in [("stage_manifest", "stage_manifest"), ("bootstrap_completion", "bootstrap"),
                      ("input_completion", "inputs"), ("primitive_report", "primitive")]:
        data = files["provenance/" + name + ".json"]
        check_hash(data, config[key]["sha256"], key)
        require(completion["admitted_input_hashes"][key] == config[key]["sha256"], "Admission hash differs")
        inputs[name] = json.loads(data)
    stage, bootstrap = inputs["stage_manifest"], inputs["bootstrap"]
    require(stage["head"] == SOURCE and stage["source_scope"] == "full"
            and stage["status"] == "source_staged_with_original_audited_wheels", "Wrong deployed public source stage")
    require(bootstrap["status"] == "packages_checked" and bootstrap["family"] == "edge"
            and bootstrap["deployment_target"] == "rtx5090" and bootstrap["python"] == config["python"]
            and bootstrap["CPU_doctor_passed"] is True and bootstrap["CPU_doctor_deferred"] is False,
            "Actual edge bootstrap/CPU doctor did not pass")
    for label in ("install_source_wheels", "pip_check", "uv_pip_check", "cpu_doctor"):
        rows = [row for row in bootstrap["commands"] if row["label"] == label]
        require(len(rows) == 1 and rows[0]["returncode"] == 0, "Bootstrap command incomplete")
        check_hash(files["provenance/bootstrap_logs/" + label + ".log"], rows[0]["log_sha256"], label)
    validate_primitive(inputs["primitive"], config, stage)
    for name, row in inputs["primitive"]["source"].items():
        relative = "scripts/qualify_sm89_fp8.py" if name == "qualify_sm89_fp8.py" else "instinctflash/runtime/" + name
        require(row["sha256"] == stage["files"][relative]["sha256"], "Primitive source differs from current installed release")
    continuity_bytes = (HERE / "installed_source_continuity.json").read_bytes()
    check_hash(continuity_bytes, CONTINUITY_SHA, "actual CPU wheel/source continuity")
    continuity = json.loads(continuity_bytes)
    require(continuity["passed"] is True and continuity["new_deployment_source_commit"] == SOURCE
            and continuity["original_installed_source_commit"] == WHEEL_SOURCE
            and continuity["new_stage_manifest_sha256"] == config["stage_manifest"]["sha256"], "Wrong CPU source continuity anchor")
    path_hashes = installed_sources(read("installed_source.json"), config, stage, bootstrap, provenance)
    check_hash(files["provenance/assets.json"], completion["assets_completion_sha256"], "public prepared assets")
    original_inputs = validate_input_rows(inputs["inputs"], json.loads(files["provenance/assets.json"]), config)
    expected_stages = ["installed_source_probe", "prepare_public_assets", "prepare_public_reproduction", "paired_public_api",
                       "serve_edge-runtime_default", "serve_edge-runtime_selected"]
    require([row["name"] for row in completion["stages"]] == expected_stages, "Missing/repeated model stages")
    for row in completion["stages"]:
        require(row["exit_code"] == 0 and type(row["pid"]) is int and str(row["start_ticks"]).isdigit()
                and row["command"][0] == config["python"] and row["elapsed_seconds"] > 0, "Failed/wrong-interpreter model stage")
        check_hash(raw(row["name"] + ".log"), row["log_sha256"], "actual stage log")
    check_hash(raw("memory_samples.jsonl"), completion["memory_samples_sha256"], "observed memory samples")
    check_hash(raw("capture/run.json"), completion["capture_run_sha256"], "actual capture run")
    plan = read("capture/plan.json")
    capture_run = read("capture/run.json")
    require(capture_run["interpreter"] == config["python"] and capture_run["cell_timeout_seconds"] == 7200,
            "Wrong actual public capture interpreter/deadline")
    for cell, attempt in zip(plan["matrix"]["cells"], capture_run["attempts"]):
        expected_command = [config["python"], "-I", "-B", "-m", "benchmarks.regression.user_e2e",
            "--matrix", RUN + "/capture/matrix.json", "--cell", cell["id"], "--output-root", RUN + "/capture",
            "--fixture", RUN + "/capture/inputs/recorded_inputs_v1.npz"]
        require(attempt["command"] == expected_command
                and attempt["optimizer_environment"] == cell["expected_optimizer_environment"],
                "Actual public capture command/environment differs")
    require(plan == load_local("expected_plan.json"), "Actual full public plan differs from audited 5090 edge FP8 plan")
    for name in ("plan.json", "matrix.json", "profiles.json", "preparation.json", "inputs/recorded_inputs_v1.npz"):
        require(raw("prepared/" + name) == raw("capture/" + name), "Prepared/captured input bytes differ")
    require(read("capture/preparation.json")["local_files_only"] is True, "Capture used nonlocal checkpoint preparation")
    with np.load(io.BytesIO(raw("capture/inputs/recorded_inputs_v1.npz")), allow_pickle=False) as fixture:
        frames = fixture["frames"].copy()
    require(frames.dtype == np.uint8 and frames.ndim == 5 and frames.shape[:2] == (13, 3) and frames.shape[-1] == 3,
            "Recorded public frame fixture differs")
    guidance_values = []
    for cell in plan["matrix"]["cells"]:
        receipt = read("capture/" + cell["receipt"])
        hardware(receipt["hardware"])
        require(receipt["interpreter"] == config["python"] and receipt["torch"] == config["expected_torch_runtime"]
                and Path(receipt["checkpoint_snapshot"]).name == plan["checkpoint"]["revision"], "Wrong model interpreter/Torch/checkpoint")
        require(receipt["numeric_environment"] == {"matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False},
                "Actual numerical environment differs")
        require(receipt["observed_nfe_before"] == receipt["observed_nfe_after"] == {'prefix': 1, 'action': 4}
                and receipt["default_schedule"] == {'prefix': 1, 'action': 4}, "Loaded checkpoint schedule differs")
        guidance_values.append(receipt["guidance"])
        backend = receipt["backend_stats"]
        native_stats = (backend if cell["arm"] == "eager_native" else
                        backend["stats"]["native_backend"] if cell["arm"] == "runtime_selected" else backend["stats"])
        require(native_stats["action_chunk_size"] == 32 and native_stats["action_steps"] == 4
                and native_stats["guidance"] == 3.0 and native_stats["conditioning_fps"] == 15.0,
                "Actual native DROID full action/CFG settings differ")
        if cell["arm"] != "eager_native":
            policy = receipt["execution_policy"]
            options = cell["expected_runtime_kwargs"]
            require(policy["nfe"] == cell["effective_schedule"]["nfe"]
                    and policy["precision"] == options.get("precision", "native")
                    and policy["tier_ceiling"] == options.get("tier_ceiling", "bitexact"),
                    "Actual Runtime execution policy differs")
        validate_requests(receipt, frames)
        sources = receipt["sources"]
        require(bool(sources) and all(path_hashes.get(path) == value for path, value in sources.items()), "Actual loaded source is outside pinned installed closure")
        for suffix in ("benchmarks/regression/hardware.py", "cosmos_framework/scripts/action_policy_server_robolab.py"):
            require(any(path.endswith("/" + suffix) for path in sources), "Native capture/loader source proof missing")
        if cell["arm"] == "eager_native":
            require(any(path.endswith("/benchmarks/regression/native_reference.py") for path in sources), "Native reference source missing")
        if cell["arm"] == "runtime_selected":
            require(receipt["precision"] == "fp8" and bool(receipt["e4m3_tensors"]), "Missing actual E4M3 weight proof")
            stats = receipt["backend_stats"]
            require(stats["status"] == "available" and stats["backend"] == "EngineBackend",
                    "Actual FP8 backend statistics unavailable or wrong backend")
            recipe = stats["stats"]["fp8_recipe"]
            require(recipe["executor"] == "sm120_torch_fp8" and recipe["capability"] == [12, 0]
                    and recipe["recipe_id"] == "sm120_cosmos3_policy_attention_qkv_v1_dense_mlp" and recipe["projections"], "Wrong actual FP8 recipe identity")
        else:
            require(receipt["precision"] == "native", "Native/default precision changed")
    require(set(guidance_values) == {"{'action': 'cfg'}"}, "Paired native guidance differs from the fixed family contract")
    independent = capture_audit.audit(path, "edge")
    require(independent["default_bitexact"] is True, "Default differs from native complete action bytes")
    require(len(independent["comparisons"]) == 2, "Missing/extra primary comparisons")
    require({(r["baseline"], r["candidate"]) for r in independent["comparisons"]}
            == {("edge-eager_native", "edge-runtime_default"), ("edge-eager_native", "edge-runtime_selected")},
            "Wrong primary comparison pairs")
    require(set(completion["serve_receipts"]) == {"edge-runtime_default", "edge-runtime_selected"}, "Both actual WebSockets required")
    websocket_rows = []
    for cell in plan["matrix"]["cells"][1:]:
        ref = completion["serve_receipts"][cell["id"]]
        name = "serve_" + cell["id"]
        require(ref["path"] == RUN + "/" + name + "/receipt.json", "WebSocket output path differs")
        row = serving(files, PREFIX, name, ref["sha256"], cell, plan["checkpoint"],
                      independent["plan_sha256"], plan["fixture_sha256"], GPU)
        ws = read(name + "/receipt.json")
        require(ws["interpreter"] == config["python"], "WebSocket used a different interpreter")
        seed = None if cell["arm"] == "runtime_selected" else 9173
        require(ws["seed"] == seed, "Native-serving constructor seed/FP8 unseeded scope differs")
        for i, call in enumerate(ws["calls"]):
            request = observation(frames, i)
            request["prompt"] = prompt(i // 3)
            require(call["request_sha256"] == request_hash(request), "Actual WebSocket fixture/reset sequence differs")
        websocket_rows.append(row)
    require(websocket_rows[0]["request_signatures"] == websocket_rows[1]["request_signatures"], "WebSocket inputs differ across arms")
    require("torch" not in sys.modules, "CPU auditor imported Torch")
    return {**independent, "schema": "instinctflash.rtx5090_edge_independent_audit.v1", "target": TARGET,
        "GPU_uuid": GPU, "source_commit": SOURCE, "original_wheel_source_commit": WHEEL_SOURCE,
        "source_continuity_sha256": CONTINUITY_SHA, "worker": identity, "completion_sha256": binding["completion_sha256"],
        "config_sha256": binding["config_sha256"], "archive_sha256": archive_sha, "installed_source_members": 432,
        "original_inputs": original_inputs, "websockets": websocket_rows,
        "vendor_cpu_provenance_sha256": launch["vendor_provenance_sha256"],
        "vendor_source_scope": "Pre-model CPU wheel/member check; actual public scanner remains limited to its original module prefixes", "GPU_used_by_auditor": False,
        "Torch_imported_by_auditor": False, "task_quality_validated": False, "minimum_RAM_certified": False,
        "scope": "Actual 5090 saved arrays/timings, fixed input/RNG/reset/full action blocks, exact installed source and two WebSockets; no recapture or task-success claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--launch-binding", required=True)
    parser.add_argument("--vendor-provenance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preparation = load_local("preparation_completion.json")
    for name, reference in preparation["files"].items():
        check_hash((HERE / name).read_bytes(), reference["sha256"], "frozen auditor input " + name)
    output = Path(args.output)
    require(not output.exists(), "Refusing to overwrite audit evidence")
    try:
        launch = json.loads(Path(args.launch_binding).read_bytes())
        provenance_bytes = Path(args.vendor_provenance).read_bytes()
        check_hash(provenance_bytes, launch["vendor_provenance_sha256"], "actual independently verified vendor provenance")
        result = audit(args.archive, json.loads(Path(args.binding).read_bytes()), launch, json.loads(provenance_bytes))
        result["actual_launch_binding_sha256"] = digest(Path(args.launch_binding).read_bytes())
    except Exception as error:
        failure = output.with_suffix(output.suffix + ".failure.json")
        with failure.open("x") as stream:
            json.dump({"passed": False, "GPU_used_by_auditor": False, "error_type": type(error).__name__,
                       "error": str(error), "requested_archive": args.archive}, stream, indent=2)
            stream.write("\n")
        raise
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "metrics": result["metrics"], "default_bitexact": result["default_bitexact"]}))


if __name__ == "__main__":
    main()
