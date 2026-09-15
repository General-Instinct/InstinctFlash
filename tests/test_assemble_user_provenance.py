"""Study sidecar assembly is local, deterministic, and preserves input bytes."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "eval/user_e2e_2026-09-14/assemble_provenance.py"
SPEC = importlib.util.spec_from_file_location("assemble_user_provenance", SCRIPT)
ASSEMBLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLER)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_local_files(root):
    identifiers = [f"{family}-{arm}" for family in ("pi05", "vla4", "vla2", "groot", "edge", "nano", "va", "dreamzero")
                   for arm in ("eager_native", "runtime_default", "runtime_selected")]
    identifiers += ["va-2v4a-native", "va-2v4a-fp8", "dreamzero-dynamic-native", "dreamzero-dynamic-fp8",
                    "vla4-runtime_update", "vla2-runtime_update", "vla2-runtime_numeric"]
    cells = [{"id": identifier, "receipt": f"cells/{identifier}/receipt.json"} for identifier in identifiers]
    cells[12]["receipt"] = "cells/edge-eager_native-path-recovery/receipt.json"
    final = {"schema": 1, "expected_main_cells": 24, "expected_operating_point_cells": 4,
             "expected_runtime_update_cells": 3, "cells": cells}
    save(root / "matrix_final.json", final)
    for name in ("matrix.json", "matrix_v2.json", "matrix_update.json", "matrix_update_vla2.json",
                 "installation.json", "installed_files.json", "installed_files_v2.json",
                 "installation_update.json", "installed_files_update.json"):
        save(root / name, {"artifact": name})
    for cell in cells[:15]:
        save(root / cell["receipt"], {"cell_id": cell["id"], "ok": True})
    save(root / "cells/edge-eager_native/receipt.json", {"cell_id": "edge-eager_native", "ok": False})
    return final


def test_explicit_12_16_1_2_assignments_bind_available_files_and_leave_future_hashes_null(tmp_path):
    final = setup_local_files(tmp_path)
    before = {str(path): digest(path) for path in tmp_path.rglob("*.json")}
    result = ASSEMBLER.assemble(tmp_path)
    assert [len(producer["cells"]) for producer in result["producers"]] == [12, 16, 1, 2]
    assert [producer["matrix"]["path"] for producer in result["producers"]] == ["matrix.json", "matrix_v2.json", "matrix_update.json", "matrix_update_vla2.json"]
    assert result["matrix"]["sha256"] == digest(tmp_path / "matrix_final.json")
    assert result["original_matrix"]["sha256"] == digest(tmp_path / "matrix.json")
    assert result["producers"][0]["cells"]["vla4-runtime_selected"] == digest(tmp_path / "cells/vla4-runtime_selected/receipt.json")
    assert result["producers"][1]["cells"]["edge-eager_native"] == digest(tmp_path / "cells/edge-eager_native-path-recovery/receipt.json")
    assert result["producers"][2]["cells"] == {"vla4-runtime_update": None}
    assert result["producers"][3]["cells"] == {"vla2-runtime_update": None, "vla2-runtime_numeric": None}
    assert result["producers"][3]["installation"]["sha256"] is None
    assert result["producers"][3]["source_inventories"][0]["sha256"] is None
    for producer in result["producers"]:
        assert producer["source_inventories"][0]["phase"] == "pre_execution"
        assert producer["source_inventories"][1] == {"path": "vendor_sources_post_execution.json", "sha256": None, "phase": "post_execution"}
    retained = result["bindings"][0]
    assert retained["path"] == "cells/edge-eager_native/receipt.json"
    assert retained["sha256"] == digest(tmp_path / retained["path"])
    assert retained["reason"] == result["receipt_relocations"][0]["reason"]
    assert len({key for producer in result["producers"] for key in producer["cells"]}) == len(final["cells"]) == 31
    assert before == {str(path): digest(path) for path in tmp_path.rglob("*.json")}


def test_reassembly_binds_new_receipt_without_selecting_its_self_reported_matrix(tmp_path):
    setup_local_files(tmp_path)
    initial = ASSEMBLER.assemble(tmp_path)
    receipt_path = tmp_path / "cells/vla4-runtime_update/receipt.json"
    save(receipt_path, {"matrix_sha256": digest(tmp_path / "matrix.json"), "sources": {"/not/trusted.py": "a" * 64}})
    later = ASSEMBLER.assemble(tmp_path)
    assert initial["producers"][2]["cells"]["vla4-runtime_update"] is None
    assert later["producers"][2]["cells"]["vla4-runtime_update"] == digest(receipt_path)
    assert later["producers"][2]["matrix"] == initial["producers"][2]["matrix"]
    assert later["producers"][2]["source_inventories"][1]["sha256"] is None


@pytest.mark.parametrize("problem", ["missing_cell", "duplicate", "unknown", "count"])
def test_unexpected_final_matrix_is_not_silently_reassigned(tmp_path, problem):
    final = setup_local_files(tmp_path)
    if problem == "missing_cell":
        final["cells"].pop()
    elif problem == "duplicate":
        final["cells"].append(final["cells"][0])
    elif problem == "unknown":
        final["cells"][-1]["id"] = "unplanned-candidate"
    else:
        final["expected_runtime_update_cells"] = 2
    save(tmp_path / "matrix_final.json", final)
    with pytest.raises(ValueError, match="Final matrix"):
        ASSEMBLER.assemble(tmp_path)


@pytest.mark.parametrize("output", ["matrix_final.json", "matrix.json", "installed_files.json", "installation.json",
                                    "cells/edge-eager_native/receipt.json", "cells/vla4-runtime_selected/receipt.json",
                                    "vendor_sources_post_execution.json"])
def test_sidecar_output_cannot_overwrite_bound_evidence(tmp_path, output):
    setup_local_files(tmp_path)
    before = {str(path): digest(path) for path in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="cannot replace"):
        ASSEMBLER.main(["--root", str(tmp_path), "--output", str(tmp_path / output)])
    assert before == {str(path): digest(path) for path in tmp_path.rglob("*.json")}


def test_cli_runs_from_outside_checkout_without_pythonpath(tmp_path):
    setup_local_files(tmp_path)
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "--root", str(tmp_path)], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["bound_receipts"] == 15
    sidecar = json.loads((tmp_path / "provenance.json").read_text())
    assert sidecar["producers"][0]["cells"]["pi05-eager_native"] == digest(tmp_path / "cells/pi05-eager_native/receipt.json")
