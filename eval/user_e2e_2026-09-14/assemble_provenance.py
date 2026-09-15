"""Assemble this study's explicit producer assignments from local artifacts.

Run this again after copying the terminal receipts and the independently hashed
vendor_sources_post_execution.json. Missing files receive null hashes, so the
separate user_provenance audit remains incomplete. This never reads remote
installed paths, derives inventories from receipt source hashes, or changes
matrices/receipts. Only the chosen sidecar output is replaced.

    python eval/user_e2e_2026-09-14/assemble_provenance.py
    python -m benchmarks.regression.user_provenance \
        --manifest eval/user_e2e_2026-09-14/provenance.json \
        --json-output eval/user_e2e_2026-09-14/provenance_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MAIN_ARMS = ("eager_native", "runtime_default", "runtime_selected")
INITIAL = tuple(f"{family}-{arm}" for family in ("pi05", "vla4", "vla2", "groot") for arm in MAIN_ARMS)
RECOVERY = tuple(f"{family}-{arm}" for family in ("edge", "nano", "va", "dreamzero") for arm in MAIN_ARMS) + (
    "va-2v4a-native", "va-2v4a-fp8", "dreamzero-dynamic-native", "dreamzero-dynamic-fp8")
GROUPS = (
    ("initial", INITIAL, "matrix.json", "installation.json", "installed_files.json"),
    ("path_recovery", RECOVERY, "matrix_v2.json", "installation.json", "installed_files_v2.json"),
    ("vla4_update", ("vla4-runtime_update",), "matrix_update.json", "installation_update.json", "installed_files_update.json"),
    ("vla2_update", ("vla2-runtime_update", "vla2-runtime_numeric"), "matrix_update_vla2.json",
     "installation_vla2_update.json", "installed_files_vla2_update.json"),
)
RECOVERY_REASON = "Edge eager path recovery changes only its receipt location; the failed original remains excluded and preserved."


def _sha_or_none(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError:
        return None
    return digest.hexdigest()


def assemble(root):
    """Return the sidecar without modifying any file or consulting receipt claims."""
    root = Path(root).resolve()
    final = json.loads((root / "matrix_final.json").read_text(encoding="utf-8"))
    rows = final["cells"]
    cells = {cell["id"]: cell for cell in rows}
    expected = {identifier for _, identifiers, *_ in GROUPS for identifier in identifiers}
    if len(cells) != len(rows) or set(cells) != expected:
        raise ValueError("Final matrix must contain exactly this study's 24 main, four operating-point and three update cells")
    if (final.get("expected_main_cells"), final.get("expected_operating_point_cells"),
            final.get("expected_runtime_update_cells")) != (24, 4, 3):
        raise ValueError("Final matrix count declarations must be 24/4/3")

    def reference(name, **extra):
        return {"path": name, "sha256": _sha_or_none(root / name), **extra}

    producers = []
    for name, identifiers, matrix, installation, inventory in GROUPS:
        producers.append({"id": name, "matrix": reference(matrix), "installation": reference(installation),
                          "source_inventories": [reference(inventory, phase="pre_execution"),
                              reference("vendor_sources_post_execution.json", phase="post_execution")],
                          "cells": {identifier: _sha_or_none(root / cells[identifier]["receipt"]) for identifier in identifiers}})
    return {"schema": 1, "scope": "Explicit study assignments; recorded provenance consistency only; no quality or source execution attestation",
            "original_matrix": reference("matrix.json"), "matrix": reference("matrix_final.json"),
            "receipt_relocations": [{"id": "edge-eager_native", "from": "cells/edge-eager_native/receipt.json",
                                     "to": "cells/edge-eager_native-path-recovery/receipt.json", "reason": RECOVERY_REASON}],
            "bindings": [reference("cells/edge-eager_native/receipt.json", reason=RECOVERY_REASON)],
            "producers": producers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="Defaults to ROOT/provenance.json")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = assemble(root)
    output = args.output.resolve() if args.output else root / "provenance.json"
    protected = {root / "matrix_final.json", Path(__file__).resolve()}
    protected.update((root / name).resolve() for _, _, *names in GROUPS for name in names)
    protected.add(root / "vendor_sources_post_execution.json")
    protected.update((root / cell["receipt"]).resolve()
                     for cell in json.loads((root / "matrix_final.json").read_text())["cells"])
    protected.add(root / "cells/edge-eager_native/receipt.json")
    if output in protected:
        raise ValueError("Sidecar output cannot replace a matrix, receipt, inventory, installation or this helper")
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cells": sum(len(producer["cells"]) for producer in manifest["producers"]),
                      "bound_receipts": sum(digest is not None for producer in manifest["producers"] for digest in producer["cells"].values())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
