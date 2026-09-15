"""Write measured same-arm array comparison using the exact historical matrix."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--assembler-sha256", required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name("assemble_v2.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == args.assembler_sha256
    spec = importlib.util.spec_from_file_location("historical_assembly_v2", source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert args.family in m.FAMILIES
    root = m.STUDY / "qualification" / args.family
    run = root / "run"
    matrix = m.read(run / "matrix.json")
    execution = m.read(run / "run.json")
    assert execution["status"] == execution["report_status"] == "passed"
    assert [row["cell"] for row in execution["attempts"]] == [cell["id"] for cell in matrix["cells"]]
    assert all(type(row["exit_code"]) is int and row["exit_code"] == 0 for row in execution["attempts"])
    rows = []
    for cell in matrix["cells"]:
        current_receipt = run / "cells" / cell["id"] / "receipt.json"
        old_receipt = m.historical_receipt(m.REPO / "eval/user_e2e_2026-09-14", cell)
        current, old = m.read(current_receipt), m.read(old_receipt)
        assert current["ok"] is True
        for key in ("cases", "model_id", "revision", "precision", "effective_schedule"):
            assert current[key] == old[key], (cell["id"], key)
        names = ["receipt.npz"] + (["receipt.queue.npz"] if "queue_drain" in current else [])
        for name in names:
            a_path, b_path = current_receipt.parent / name, old_receipt.parent / name
            a_hash = current["actions_sha256"] if name == "receipt.npz" else current["queue_drain"]["actions_sha256"]
            b_hash = old["actions_sha256"] if name == "receipt.npz" else old["queue_drain"]["actions_sha256"]
            fresh, historical = m.arrays(a_path, a_hash), m.arrays(b_path, b_hash)
            assert set(fresh) == set(historical) == {"actions"}
            a, b = fresh["actions"], historical["actions"]
            same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
            exact = same_shape and same_dtype and a.tobytes() == b.tobytes()
            maximum = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if same_shape else None
            rows.append({"cell": cell["id"], "file": name, "shape": list(a.shape), "dtype": str(a.dtype),
                         "historical_shape": list(b.shape), "historical_dtype": str(b.dtype),
                         "same_shape": same_shape, "same_dtype": same_dtype,
                         "historical_file": str(b_path.relative_to(m.REPO)), "historical_file_sha256": b_hash,
                         "new_file": str(a_path.relative_to(m.REPO)), "new_file_sha256": a_hash,
                         "historical_receipt": m.ref(old_receipt), "new_receipt": m.ref(current_receipt),
                         "historical_actions_match_bytes": exact, "max_abs": maximum,
                         "historical_matrix": m.ref(m.REPO / "eval/user_e2e_2026-09-14/matrix_final.json"),
                         "comparison_source_sha256": m.sha(Path(__file__)), "assembler_source_sha256": m.sha(source),
                         "scope": "Each new arm versus its declared historical arm; no task-quality claim."})
    output = root / "historical_action_comparison_v1.json"
    m.save(output, rows)
    print(json.dumps({"path": str(output), "sha256": m.sha(output), "arrays": len(rows),
                      "exact": sum(row["historical_actions_match_bytes"] for row in rows)}))


if __name__ == "__main__":
    main()
