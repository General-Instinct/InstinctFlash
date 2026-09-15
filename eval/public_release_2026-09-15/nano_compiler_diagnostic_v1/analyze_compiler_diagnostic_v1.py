"""One-arm compiler diagnostic replay; original historical gates stay exact."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def validate_completion(config, completion, diagnostic):
    require(completion["status"] == "passed" and len(completion["jobs"]) == len(config["jobs"]) == 1, "External one-arm queue did not pass")
    job, selected = completion["jobs"][0], config["jobs"][0]
    require(job["mode"] == selected["mode"] == diagnostic["mode"] == "observed", "Wrong diagnostic mode")
    require(job["argv"] == selected["argv"] and job["argv"][1:3] == ["-I", "-B"], "Executed isolated command differs")
    require(type(job["returncode"]) is int and job["returncode"] == 0, "External native process exit was not zero")
    require(job["receipt_status"] == diagnostic["status"] == "diagnostic_capture_completed", "Native result did not complete")
    require(0 < job["completed_unix"] - job["started_unix"] < 900, "Native execution exceeded its finite bound")
    require(completion["started_unix"] <= job["started_unix"] < job["completed_unix"] <= completion["completed_unix"], "External process chronology differs")


def validate_generated(root, diagnostic):
    generated = diagnostic["runtime_provenance"]["generated_python_modules"]
    manifest_path = root / "generated_code/manifest.json"
    require(sha(manifest_path) == generated["manifest"]["sha256"], "Generated manifest changed")
    manifest = read(manifest_path)
    require(manifest["status"] == generated["status"] == "captured_loaded_generated_modules", "Generated source capture did not pass")
    require(len(manifest["modules"]) == generated["modules"] > 0, "Generated Python modules missing")
    require(len(manifest["artifacts"]) == generated["files"], "Generated file inventory count differs")
    for name, entry in manifest["artifacts"].items():
        require(Path(name).name == name, "Generated artifact path escape")
        path = root / "generated_code" / name
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], "Generated artifact bytes changed")
    require(sum(row["bytes"] for row in manifest["artifacts"].values()) == generated["bytes"] == manifest["bytes"], "Generated inventory byte count differs")
    fresh = diagnostic["fresh_compiler_caches"]
    require(fresh["status"] == "created_exact_new_empty_caches" and fresh["initial_entries"] == 0, "Compiler cache was not prospectively empty")
    for module in manifest["modules"]:
        require(Path(module["original"]["path"]).is_relative_to(fresh["paths"]["TORCHINDUCTOR_CACHE_DIR"]), "Module came from a shared or old cache")
    return {"manifest": ref(manifest_path), "modules": generated["modules"],
            "selected_cached_launchers": generated["selected_cached_launchers"], "files": generated["files"],
            "bytes": generated["bytes"], "dynamic_kernel_execution_trace": False}


def read_capture(root, expected_source, binding):
    diagnostic = read(root / "diagnostic.json")
    require(diagnostic["source"]["sha256"] == expected_source and diagnostic["completed_requests"] == 25, "Diagnostic source/count differs")
    require(diagnostic["status"] == "diagnostic_capture_completed", "Diagnostic failed")
    require(sha(root / "trace.jsonl") == diagnostic["trace"]["sha256"], "Diagnostic trace changed")
    rows = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    require([r["event"] for r in rows] == list(range(diagnostic["events"])), "Trace sequence changed")
    for kind in ("request_before", "request_after", "prepared_noise", "native_seed"):
        require([r["request"] for r in rows if r["kind"] == kind] == list(range(25)), "Incomplete per-request trace")
    for kind in ("layer_before", "layer_after"):
        require([(r["request"], r["velocity"], r["layer"]) for r in rows if r["kind"] == kind] == [(0, i, layer) for i in range(8) for layer in range(36)], "Incomplete first-request layer trace")
    path = root / binding["cell"]["receipt"]
    require(sha(path) == diagnostic["native_receipt"]["sha256"], "Native receipt changed")
    native = read(path)
    require(native["ok"] is True and len(native["cases"]) == 25, "Incomplete native capture")
    require(sha(path.with_suffix(".npz")) == native["actions_sha256"], "Native actions changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        actions = archive["actions"].copy()
    require(actions.shape == (25, 32, 8) and actions.dtype == np.float32 and np.isfinite(actions).all(), "Invalid native action array")
    return diagnostic, native, rows, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior-observed", type=Path, required=True)
    parser.add_argument("--queue-config", type=Path, required=True)
    parser.add_argument("--external-completion", type=Path, required=True)
    parser.add_argument("--historical-actions", type=Path, required=True)
    parser.add_argument("--fresh-actions", type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--compiler-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(HERE / "run_compiler_diagnostic_v1.py") == args.worker_sha256, "Actual compiler diagnostic source changed")
    require(sha(HERE / "compiler_binding_v1.json") == args.compiler_binding_sha256, "Compiler binding changed")
    compiler = read(HERE / "compiler_binding_v1.json")
    require(sha(HERE / "original_binding_v1.json") == compiler["original_binding_sha256"], "Original binding changed")
    binding = read(HERE / "original_binding_v1.json")
    helper = HERE / "parent_analysis_tools_6b.py"
    require(sha(helper) == "6b94690b550dac34364d0e6ffecd1061c6b7132c033f1f364dc3c7b7b9fd0347", "Parent analysis helpers changed")
    spec = importlib.util.spec_from_file_location("original_analysis_helpers", helper)
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    current, native, rows, actions = read_capture(args.run, args.worker_sha256, binding)
    require(current["compiler_binding"]["sha256"] == args.compiler_binding_sha256, "Executed compiler binding differs")
    require(current["loaded_compiler_verification"]["status"] == "loaded_compiler_and_control_package_libraries_verified", "Loaded compiler/control binary gate failed")
    previous, old_native, old_rows, old_actions = read_capture(args.prior_observed, compiler["parent_worker_sha256"], binding)
    require(sha(args.prior_observed / "diagnostic.json") == compiler["fresh_control_diagnostic_sha256"], "Prior observed control differs")
    validate_completion(read(args.queue_config), read(args.external_completion), current)
    generated = validate_generated(args.run, current)
    for key in ("cases", "effective_schedule", "runtime_kwargs", "optimizer_environment", "model_id", "revision"):
        require(native[key] == old_native[key], f"Original native execution contract changed: {key}")
    for path, expected in ((args.historical_actions, binding["historical_selected_archive_sha256"]), (args.fresh_actions, binding["fresh_selected_archive_sha256"])):
        require(sha(path) == expected, "Frozen comparison action archive changed")
    with np.load(args.historical_actions, allow_pickle=False) as archive:
        historical = archive["actions"].copy()
    with np.load(args.fresh_actions, allow_pickle=False) as archive:
        fresh = archive["actions"].copy()
    fields = {"request_before": ["input", "rng", "flags"], "native_seed": ["value"],
              "prepared_noise": ["tokens", "noise", "condition_reference", "condition_mask", "rng_before", "rng_after"],
              "velocity_before": ["noise", "timestep", "tokens", "rng"], "velocity_after": ["output", "rng"],
              "layer_before": ["values"], "layer_after": ["values"]}
    result = {"schema": "instinctflash.nano_compiler_diagnostic_comparison.v1", "status": "diagnostic_comparison_complete",
              "source": ref(__file__), "worker": ref(HERE / "run_compiler_diagnostic_v1.py"),
              "compiler_binding": ref(HERE / "compiler_binding_v1.json"), "current": ref(args.run / "diagnostic.json"),
              "previous_control": ref(args.prior_observed / "diagnostic.json"), "external_completion": ref(args.external_completion),
              "queue_config": ref(args.queue_config), "generated_artifacts": generated,
              "actions": {"corrected_vs_historical": tools.action_delta(actions, historical),
                          "corrected_vs_fresh": tools.action_delta(actions, fresh),
                          "corrected_vs_prior_observed": tools.action_delta(actions, old_actions)},
              "traces_prior_observed_then_corrected": {kind: tools.compare_events(old_rows, rows, kind, names) for kind, names in fields.items()},
              "historical_action_bytes_recovered_on_tested_inputs": tools.action_delta(actions, historical)["exact_bytes"],
              "task_quality_certified": False, "normal_benchmark_or_serving_qualified": False,
              "limitations": ["Exact source/shape/schedule/compiler and external process gates remain enforced; action differences are reported without tolerance waivers.",
                              "Trace helper field labels observed/conditioning_bypass mean prior observed/corrected compiler here; this run contains no bypass treatment.",
                              "The corrected run uses newly empty compiler caches; the prior control reused its existing cache. Compiler package and cache state are both explicit, so causal attribution must inspect emitted code and first differing tensors.",
                              "Instrumented diagnostic timing is not a benchmark. Separate additive Edge/Nano paired and WebSocket qualifications remain required."]}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "actions": result["actions"]}))


if __name__ == "__main__":
    main()
