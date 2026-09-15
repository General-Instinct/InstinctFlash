"""CPU replay of the retained Nano historical discrepancy; no acceptance override."""

import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
QUALIFICATION = REPO / "eval/public_release_2026-09-15/qualification/nano"
OLD = REPO / "eval/user_e2e_2026-09-14/cells/nano-runtime_selected/receipt.json"
NEW = QUALIFICATION / "run/cells/nano-runtime_selected/receipt.json"
RECEIPT_HASHES = (
    "23f98fe419c72a0453c7be3d193be14c07fd612b80cd3a5d71e38406baba976e",
    "6f335e09bffc69dcfc26754430a3bfb7bc76f0d4c1a4f8af6e66642508bfec50",
)
PREFIXES = ("instinctflash/", "cosmos3_iwm/", "cosmos_framework/", "benchmarks/")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path), "sha256": sha(path)}


def selected_sources(receipt):
    result = {}
    excluded = []
    for path, digest in receipt["sources"].items():
        keys = [prefix + path.split("/" + prefix, 1)[1] for prefix in PREFIXES if "/" + prefix in path]
        if not keys:
            excluded.append({"path": path, "sha256": digest})
            continue
        assert len(keys) == 1 and keys[0] not in result
        result[keys[0]] = digest
    return result, excluded


def stats(receipt):
    backend = receipt["backend_stats"]["stats"]
    cache = backend["conditioning_cache"]
    graphs = cache["graph_stats"]
    return {
        "conditioning_cache": {k: v for k, v in cache.items() if k != "graph_stats"},
        "conditioning_graph_entries": len(graphs),
        "conditioning_graph_sums": {
            key: sum(entry.get(key, 0) for entry in graphs)
            for key in ("captures", "checks", "replays", "memory_bypasses")
        },
        "layer_graphs": backend["layer_graphs"],
        "generation_regions": backend["generation_regions"],
        "numeric_attention": backend["numeric_attention"],
        "timestep_cache": backend["timestep_cache"],
        "peak_allocated_bytes": receipt["peak_allocated_bytes"],
    }


def main():
    for path, expected in zip((OLD, NEW), RECEIPT_HASHES):
        assert sha(path) == expected, "Frozen receipt changed"
    old, new = [json.loads(path.read_text()) for path in (OLD, NEW)]
    arrays = []
    for path, receipt in zip((OLD, NEW), (old, new)):
        assert sha(path.with_suffix(".npz")) == receipt["actions_sha256"]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            assert archive.files == ["actions"]
            arrays.append(archive["actions"])
    a, b = arrays
    assert a.shape == b.shape == (25, 32, 8) and a.dtype == b.dtype == np.float32
    assert np.isfinite(a).all() and np.isfinite(b).all()
    same_keys = ("cases", "model_id", "revision", "effective_schedule", "applied_passes", "numeric_environment", "optimizer_environment", "torch")
    assert all(old[key] == new[key] for key in same_keys)
    left, excluded_left = selected_sources(old)
    right, excluded_right = selected_sources(new)
    assert left.keys() == right.keys()
    diff = b.astype(np.float64) - a.astype(np.float64)
    rows = [{"i": i, "seed": old["cases"][i]["seed"],
             "exact_bytes": a[i].tobytes() == b[i].tobytes(),
             "max_abs": float(np.abs(row).max()),
             "rmse": float(np.square(row).mean() ** 0.5)} for i, row in enumerate(diff)]
    result = {
        "schema": "instinctflash.nano_historical_numerical_drift_review.v1",
        "status": "historical_selected_mismatch_unresolved",
        "source": ref(Path(__file__).resolve()),
        "receipts": {"historical": ref(OLD), "new": ref(NEW)},
        "historical_parity": ref(QUALIFICATION / "historical_action_comparison_v1.json"),
        "new_summary": ref(Path(__file__).parent / "nano_completed_v1/summary.json"),
        "action_arrays": {"historical": ref(OLD.with_suffix(".npz")), "new": ref(NEW.with_suffix(".npz"))},
        "identical_declared_contract_fields": list(same_keys),
        "source_inventory": {
            "normalization": "Exact package-relative paths in four declared runtime/vendor namespaces; collisions rejected",
            "common_entries": len(left),
            "matching_entries": sum(left[key] == right[key] for key in left),
            "differences": [{"path": key, "historical_sha256": left[key], "new_sha256": right[key]} for key in sorted(left) if left[key] != right[key]],
            "excluded_other_package_entries": {"historical": excluded_left, "new": excluded_right},
        },
        "historical_execution": stats(old),
        "new_execution": stats(new),
        "difference": {"shape": list(a.shape), "dtype": str(a.dtype), "exact_bytes": a.tobytes() == b.tobytes(),
                       "max_abs": float(np.abs(diff).max()), "rmse": float(np.square(diff).mean() ** 0.5),
                       "mean_abs": float(np.abs(diff).mean()), "different_values": int(np.count_nonzero(diff)),
                       "total_values": int(diff.size), "requests": rows},
        "interpretation": "Graph admission/memory statistics differ, but their causal contribution is not established. No GPU recapture, threshold change, or source modification was performed.",
        "historical_equivalence_proven": False,
        "task_quality_certified": False,
    }
    output = QUALIFICATION / "historical_drift_review_v1.json"
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(ref(output)))


if __name__ == "__main__":
    main()
