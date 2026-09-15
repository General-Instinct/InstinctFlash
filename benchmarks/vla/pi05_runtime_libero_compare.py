"""Verify complete paired Runtime campaigns before reporting their observed outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pi05_runtime_libero_campaign import frozen_jobs, validate_identity
from .util import ConfigurationError, sha256_file, sha256_json, write_json_atomic


def require(condition, message):
    if not condition:
        raise ConfigurationError(message)


def local_file(root, name):
    path = Path(name)
    require(len(path.parts) == 1 and not path.is_absolute() and path.name not in {".", ".."},
            "unsafe artifact path")
    return root / path


def verify_episode(root, manifest, job):
    from instinctflash.serving.msgpack_numpy import unpackb

    result_path = local_file(root, job["result"])
    array_path = result_path.with_suffix(".npz")
    trace_root = result_path.with_suffix(".trace")
    for path, key in ((result_path, "result_sha256"), (array_path, "actions_sha256"),
                      (trace_root / "trace.json", "trace_sha256")):
        require(sha256_file(path) == job[key], f"changed artifact: {path}")
    result = json.loads(result_path.read_text())
    require(result.get("ok") is True and type(result.get("success")) is bool,
            "missing or invalid episode outcome")
    require(type(result.get("steps")) is int and 1 <= result["steps"] <= 520,
            "invalid episode length")
    for key in ("task", "seed", "success", "steps"):
        require(result[key] == job[key], f"job/result mismatch: {key}")
    for key in ("identity", "simulator_identity", "scene_sha256"):
        require(result[key] == manifest[key], f"episode/arm mismatch: {key}")
    require(result["settle_steps"] == 10 and result["settle_action"] == [0, 0, 0, 0, 0, 0, -1]
            and result["camera_size"] == [360, 360], "changed environment protocol")
    shapes = {"agentview_image": (360, 360, 3), "robot0_eye_in_hand_image": (360, 360, 3),
              "robot0_eef_pos": (3,), "robot0_eef_quat": (4,), "robot0_gripper_qpos": (2,)}
    with np.load(array_path, allow_pickle=False) as arrays:
        actions = arrays["actions"].copy()
        initial = {key: arrays["initial_" + key].copy() for key in shapes}
    require(actions.shape == (result["steps"], 7) and np.isfinite(actions).all(),
            "invalid retained actions")
    for key, value in initial.items():
        require(value.shape == shapes[key] and np.isfinite(value).all(), "invalid initial state")
    trace = json.loads((trace_root / "trace.json").read_text())
    require(trace.get("complete") is True and trace.get("schema_version") == 1
            and trace.get("identity") == manifest["identity"], "invalid trace identity/completion")
    require(len(trace["calls"]) == result["steps"] + 1, "missing/extra trace calls")
    prompt = None
    for index, call in enumerate(trace["calls"]):
        values = {}
        for kind in ("request", "response"):
            path = local_file(trace_root, call[kind]["path"])
            require(sha256_file(path) == call[kind]["sha256"], "changed wire packet")
            values[kind] = unpackb(path.read_bytes())
        request, response = values["request"], values["response"]
        if index == 0:
            digest = sha256_json(manifest["identity"])
            require(request.get("reset") is True, "missing explicit episode reset")
            for value in (request, response):
                require(value.get("benchmark_seed") == job["seed"]
                        and value.get("benchmark_identity_sha256") == digest,
                        "reset seed/identity acknowledgement mismatch")
            prompt = request["prompt"]
            continue
        require(not request.get("reset") and request.get("prompt") == prompt,
                "unexpected mid-episode reset or prompt change")
        raw = request["libero_observation"]
        for key, shape in shapes.items():
            value = np.asarray(raw[key])
            require(value.shape == shape and np.isfinite(value).all(), "invalid wire observation")
            if index == 1:
                require(value.dtype == initial[key].dtype and value.tobytes() == initial[key].tobytes(),
                        "retained initial state differs from first wire observation")
        require(np.array_equal(response["action"], actions[index - 1]), "wire/retained action mismatch")
    return result, initial


def compare(native_root, fp8_root):
    manifests = {}
    for mode, root in (("native", native_root), ("fp8", fp8_root)):
        manifest = json.loads((root / "campaign.json").read_text())
        require(manifest.get("status") == "complete"
                and manifest.get("protocol") == "pi05-public-libero-50-campaign-v1",
                f"{mode} campaign is incomplete or incompatible")
        validate_identity(manifest["identity"])
        require(manifest["identity"]["precision"] == mode, "precision arm mislabeled")
        require(sha256_file(root / "scenes.json") == manifest["scene_sha256"], "changed frozen scenes")
        expected = frozen_jobs(json.loads((root / "scenes.json").read_text()))
        require([{k: j[k] for k in ("task", "seed", "key")} for j in manifest["jobs"]] == expected,
                "missing, duplicate or substituted scene jobs")
        require(all(j["status"] == "complete" and j["exit_code"] == 0 for j in manifest["jobs"]),
                "campaign contains unsuccessful processes")
        manifests[mode] = manifest
    native, fp8 = manifests["native"], manifests["fp8"]
    for key in ("scene_sha256", "simulator_identity", "driver_sources"):
        require(native[key] == fp8[key], f"paired arm mismatch: {key}")
    require({k: v for k, v in native["identity"].items() if k != "precision"}
            == {k: v for k, v in fp8["identity"].items() if k != "precision"},
            "paired runtime identities differ beyond precision")
    rows = []
    for nj, fj in zip(native["jobs"], fp8["jobs"]):
        nr, ni = verify_episode(native_root, native, nj)
        fr, fi = verify_episode(fp8_root, fp8, fj)
        require(all(ni[k].dtype == fi[k].dtype and ni[k].tobytes() == fi[k].tobytes() for k in ni),
                f"unpaired initial observations: {nj['key']}")
        rows.append({"task": nj["task"], "seed": nj["seed"], "native_success": nr["success"],
                     "fp8_success": fr["success"], "native_steps": nr["steps"], "fp8_steps": fr["steps"]})
    ns, fs = (sum(r[f"{mode}_success"] for r in rows) for mode in ("native", "fp8"))
    return {"scope": "20 paired-scene screen, not non-inferiority, zero-loss or real-time certification",
            "pairs": len(rows), "native_successes": ns, "fp8_successes": fs,
            "observed_success_delta_pp": 100 * (fs - ns) / len(rows),
            "native_only_successes": sum(r["native_success"] and not r["fp8_success"] for r in rows),
            "fp8_only_successes": sum(r["fp8_success"] and not r["native_success"] for r in rows),
            "initial_observations_byte_equal": True, "all_wire_and_action_artifacts_verified": True,
            "manifest_sha256": {m: sha256_file(r / "campaign.json") for m, r in
                                (("native", native_root), ("fp8", fp8_root))}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to replace a comparison")
    write_json_atomic(args.output, compare(args.native, args.fp8))


if __name__ == "__main__":
    main()
