"""Check unchanged actions and realized graph reuse before publishing prompt fixes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


UPDATES = {
    "vla4-runtime_update": ("vla4-runtime_selected", "b",
        "serving/flash_rt/frontends/torch/vla4b_thor.py",
        {"prompt_updates": 25, "prompt_buffer_reuses": 24, "prompt_graph_captures": 1, "replays": 25}),
    "vla2-runtime_update": ("vla2-runtime_selected", "c",
        "serving/flash_rt/frontends/torch/vla2_thor.py",
        {"prompt_updates": 25, "prompt_buffer_reuses": 24, "language_action_captures": 1,
         "language_action_prompt_reuses": 24, "expert_calibrations": 25, "replays": 25}),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _receipt_rows(audit, name, expected_ids):
    records = audit.get("cells")
    require(isinstance(records, list) and len(records) == 31, f"{name} must contain 31 receipt bindings")
    rows = {}
    for row in records:
        require(isinstance(row, dict), f"{name} receipt binding must be an object")
        identifier = row.get("id")
        require(isinstance(identifier, str) and identifier in expected_ids and identifier not in rows,
                f"{name} receipt IDs must match the complete matrix exactly once")
        require(row.get("status") == "passed", f"{name} receipt binding is not passed: {identifier}")
        receipt = row.get("receipt")
        require(isinstance(receipt, dict), f"{name} receipt binding is missing: {identifier}")
        digest = receipt.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest),
                f"{name} receipt SHA-256 is invalid: {identifier}")
        rows[identifier] = row
    require(set(rows) == expected_ids, f"{name} receipt IDs differ from the complete matrix")
    return rows


def bind_report_receipts(report, provenance, expected_ids):
    """Join both complete audits before trusting provenance for measured rows."""
    require(len(expected_ids) == 31, "final matrix must contain 31 unique cells")
    rows = _receipt_rows(report, "report", expected_ids)
    verified = _receipt_rows(provenance, "provenance", expected_ids)
    for identifier, row in rows.items():
        require(row["receipt"]["sha256"] == verified[identifier]["receipt"]["sha256"],
                f"report/provenance receipt SHA-256 mismatch: {identifier}")
    return rows


def check_update(old, new, before, after, *, source_suffix, source_sha256, counters):
    errors = []
    for name in ("family", "precision", "device", "torch", "model_id", "revision", "cases",
                 "runtime_kwargs", "optimizer_environment", "effective_schedule", "guidance"):
        if name not in old or old.get(name) != new.get(name):
            errors.append(f"paired {name} changed or missing")
    if old.get("precision") != "fp8" or not old.get("ok") or not new.get("ok"):
        errors.append("both original and updated FP8 captures must have passed")
    matching = [value for path, value in new.get("sources", {}).items() if path.endswith(source_suffix)]
    if matching != [source_sha256]:
        errors.append("actual loaded frontend source does not match the frozen candidate")
    arrays_match = before.shape == after.shape and before.dtype == after.dtype
    finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
    exact = bool(arrays_match and finite and before.tobytes() == after.tobytes())
    if not exact or before.shape[0] != 25:
        errors.append("all 25 full public actions must match finite bytes exactly")
    stats = new.get("graph_stats", {})
    if stats.get("captured") is not True:
        errors.append("updated language/action graph is not active")
    for key, expected in counters.items():
        if type(stats.get(key)) is not int or stats[key] != expected:
            errors.append(f"unexpected realized {key}: {stats.get(key)!r}, expected {expected}")
    return dict(status="failed" if errors else "passed", exact_action_bytes=exact,
        action_shape=list(after.shape), action_dtype=str(after.dtype), graph_stats=stats,
        loaded_frontend_sha256=matching, errors=errors, task_quality_validated=False,
        scope="Same FP8 execution, requests and calibration; setup reuse only; no native/FP8 quality equivalence")


def build_gate(root):
    root = Path(root)
    report_path, provenance_path = root / "report.json", root / "provenance_report.json"
    report, provenance = (json.loads(p.read_text()) for p in (report_path, provenance_path))
    if report["status"] != "passed" or provenance["status"] != "passed":
        raise ValueError("Complete protocol and provenance audits must pass first")
    matrix_sha = sha(root / "matrix_final.json")
    require(report["matrix"]["sha256"] == matrix_sha, "report refers to a different matrix")
    require(any(item["sha256"] == matrix_sha for item in provenance["artifacts"]),
            "provenance did not bind the final matrix")
    matrix = json.loads((root / "matrix_final.json").read_text())
    cells = {cell["id"]: cell for cell in matrix["cells"]}
    require(len(cells) == len(matrix["cells"]) == 31, "final matrix must contain 31 unique cells")
    rows = bind_report_receipts(report, provenance, set(cells))
    gates = {}

    def load(identifier):
        path = root / cells[identifier]["receipt"]
        receipt = json.loads(path.read_text())
        require(sha(path) == rows[identifier]["receipt"]["sha256"], "receipt changed after validation")
        archive = path.with_suffix(".npz")
        require(sha(archive) == receipt["actions_sha256"], "action archive changed after capture")
        with np.load(archive, allow_pickle=False) as data:
            actions = data["actions"].copy()
        return receipt, actions

    for identifier, (baseline, version, source, counters) in UPDATES.items():
        old, before = load(baseline)
        new, after = load(identifier)
        manifest_path = root / f"source_manifest_{version}.json"
        install_path = root / ("installation_update.json" if version == "b"
                               else "installation_vla2_update.json")
        installation = json.loads(install_path.read_text())
        require(sha(manifest_path) == installation["candidate_manifest_sha256"],
                "candidate source manifest differs from the installed version")
        require(any(item["sha256"] == sha(install_path) for item in provenance["artifacts"]),
                "provenance did not bind this candidate installation")
        manifest = json.loads(manifest_path.read_text())
        gate = check_update(old, new, before, after,
            source_suffix="/" + source.removeprefix("serving/"),
            source_sha256=manifest["files"][source]["sha256"], counters=counters)
        gate.update(receipt_sha256=rows[identifier]["receipt"]["sha256"],
            baseline_receipt_sha256=rows[baseline]["receipt"]["sha256"])
        gates[identifier] = gate
    identifier = "vla2-runtime_numeric"
    receipt, _ = load(identifier)
    comparison = next(c for c in report["comparisons"]
        if c["baseline"] == "vla2-eager_native" and c["candidate"] == identifier)
    require(comparison["status"] == "passed" and comparison["same_sampling_policy"],
            "NUMERIC comparison did not preserve the sampling policy")
    require(receipt["precision"] == "native" and receipt["runtime_kwargs"] == {
        "device": "cuda:0", "tier_ceiling": "numeric"}, "unexpected NUMERIC execution request")
    captured = receipt.get("graph_stats", {}).get("captured") is True
    gates[identifier] = dict(status="passed" if captured else "inactive",
        receipt_sha256=rows[identifier]["receipt"]["sha256"],
        graph_stats=receipt.get("graph_stats"), actions=comparison["actions"],
        scope="Existing native NUMERIC capture active on paired requests; numerical SCREEN only",
        task_quality_validated=False)
    return dict(schema=1, status="passed" if all(gates[k]["status"] == "passed" for k in UPDATES) else "failed",
        report_sha256=sha(report_path), provenance_report_sha256=sha(provenance_path),
        validator_sha256=sha(__file__), gates=gates, task_quality_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = build_gate(args.root)
    (args.root / "prompt_update_gate.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value["status"] for key, value in report["gates"].items()}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
