import importlib.util
import json
from pathlib import Path
import shutil

import pytest


PATH = Path(__file__).with_name("assemble_v2.py")
spec = importlib.util.spec_from_file_location("evidence_assembly_v2", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
HISTORY = m.REPO / "eval/user_e2e_2026-09-14"
CELL = m.read(m.STUDY / "qualification/edge/run/matrix.json")["cells"][0]


def test_exact_historical_matrix_selects_successful_recovery_and_preserves_failed_original():
    receipt = m.historical_receipt(HISTORY, CELL)
    assert receipt.parent.name == "edge-eager_native-path-recovery"
    assert m.read(receipt)["ok"] is True
    assert m.read(HISTORY / "cells/edge-eager_native/receipt.json")["ok"] is False
    assert m.sha(PATH.with_name("assemble.py")) == m.PARENT_ASSEMBLER_SHA


def test_historical_matrix_path_mutation_rejected(tmp_path):
    matrix = m.read(HISTORY / "matrix_final.json")
    next(row for row in matrix["cells"] if row["id"] == CELL["id"])["receipt"] = "cells/edge-eager_native/receipt.json"
    (tmp_path / "matrix_final.json").write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="final matrix changed"):
        m.historical_receipt(tmp_path, CELL)


def test_missing_declared_recovery_never_falls_back_to_unbound_success(tmp_path):
    shutil.copyfile(HISTORY / "matrix_final.json", tmp_path / "matrix_final.json")
    alternative = tmp_path / "cells/edge-eager_native/receipt.json"
    alternative.parent.mkdir(parents=True)
    shutil.copyfile(HISTORY / "cells/edge-eager_native-path-recovery/receipt.json", alternative)
    with pytest.raises(FileNotFoundError):
        m.historical_receipt(tmp_path, CELL)


def test_same_id_with_changed_sampling_contract_rejected():
    cell = dict(CELL, effective_schedule={"sampler": "sde", "steps": 1})
    with pytest.raises(ValueError, match="cell contract differs"):
        m.historical_receipt(HISTORY, cell)
