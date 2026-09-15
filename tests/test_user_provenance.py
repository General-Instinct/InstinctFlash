"""Pure CPU integrity tests for preserved-matrix and installed-source evidence."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.regression.user_provenance import ADAPTERS, audit_provenance, main


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(root, name, **extra):
    return {"path": name, "sha256": digest(root / name), **extra}


def setup_study(root):
    families = ("pi05", "vla4", "vla2", "groot", "edge", "nano", "va", "dreamzero")
    cells = []
    for family in families:
        for arm in ("eager_native", "runtime_default", "runtime_selected"):
            identifier = f"{family}-{arm}"
            cells.append({"id": identifier, "family": family, "arm": arm,
                          "receipt": f"cells/{identifier}.json", "model_id": f"owner/{family}",
                          "revision": "a" * 40, "comparison_group": family, "experimental": False})
    for family in ("va", "dreamzero"):
        for precision in ("native", "fp8"):
            identifier = f"{family}-extra-{precision}"
            cells.append({"id": identifier, "family": family, "arm": "operating_point",
                          "receipt": f"cells/{identifier}.json", "model_id": f"owner/{family}",
                          "revision": "a" * 40, "comparison_group": f"{family}-extra", "experimental": True})
    original = {"schema": 1, "expected_main_cells": 24, "expected_families": list(families),
                "quality_certified": False, "protocol": {"warmup": 5, "measured": 20}, "cells": cells}
    save(root / "matrix.json", original)
    v2 = copy.deepcopy(original)
    edge = next(cell for cell in v2["cells"] if cell["id"] == "edge-eager_native")
    old_receipt = edge["receipt"]
    edge["receipt"] = "cells/edge-recovery.json"
    save(root / "matrix_v2.json", v2)
    final = copy.deepcopy(v2)
    update = copy.deepcopy(next(cell for cell in final["cells"] if cell["id"] == "vla4-runtime_selected"))
    update.update(id="vla4-runtime_update", arm="runtime_update", receipt="cells/vla4-runtime_update.json",
                  baseline_cell="vla4-runtime_selected")
    final["cells"].append(update)
    final.update(expected_operating_point_cells=4, expected_runtime_update_cells=1)
    save(root / "matrix_final.json", final)
    installation = {"schema": 1, "ok": True, "environments": {}}
    installed, external, source_sets = {}, {}, {}
    for family in families:
        site = f"/installed/{family}/lib/python3.12/site-packages"
        installation["environments"][family] = {"python": f"/installed/{family}/bin/python", "overlay_site": site}
        names = ["instinctflash/__init__.py", "instinctflash/runtime/facade.py"]
        names.append(f"{ADAPTERS[family]}/adapter.py" if family in ADAPTERS else "instinctflash/adapters/lingbot_va.py")
        sources = {f"{site}/{name}": hashlib.sha256(name.encode()).hexdigest() for name in names}
        installed.update(sources)
        installed[f"{site}/benchmarks/regression/user_e2e.py"] = "c" * 64
        external[f"/vendor/{family}/model.py"] = "e" * 64
        source_sets[family] = {**sources, f"/vendor/{family}/model.py": "e" * 64}
    save(root / "installation.json", installation)
    save(root / "installed_files.json", installed)
    save(root / "vendor_files.json", external)
    updated = dict(installed)
    changed_source = "/installed/vla4/lib/python3.12/site-packages/instinctflash/runtime/facade.py"
    updated[changed_source] = "f" * 64
    save(root / "installed_files_update.json", updated)
    producers = []
    for name, matrix_name in (("original", "matrix.json"), ("recovery", "matrix_v2.json"), ("update", "matrix_final.json")):
        producers.append({"id": name, "matrix": reference(root, matrix_name),
                          "installation": reference(root, "installation.json"),
                          "source_inventories": [reference(root, "installed_files_update.json" if name == "update" else "installed_files.json", phase="pre_execution"),
                                                 reference(root, "vendor_files.json", phase="post_execution")], "cells": {}})
    for cell in final["cells"]:
        producer = producers[2] if cell["arm"] == "runtime_update" else producers[1] if cell["family"] == "edge" else producers[0]
        sources = dict(source_sets[cell["family"]])
        if producer["id"] == "update":
            sources[changed_source] = updated[changed_source]
        receipt = {"schema": 1, "ok": True, "cell_id": cell["id"], "family": cell["family"], "arm": cell["arm"],
                   "model_id": cell["model_id"], "revision": cell["revision"], "matrix_sha256": producer["matrix"]["sha256"],
                   "interpreter": installation["environments"][cell["family"]]["python"], "sources": sources}
        save(root / cell["receipt"], receipt)
        producer["cells"][cell["id"]] = digest(root / cell["receipt"])
    # An excluded failed original remains untouched and is never selected.
    save(root / old_receipt, {"ok": False, "error": "old path configuration"})
    manifest = {"schema": 1, "original_matrix": reference(root, "matrix.json"), "matrix": reference(root, "matrix_final.json"),
                "receipt_relocations": [{"id": "edge-eager_native", "from": old_receipt, "to": edge["receipt"],
                                         "reason": "Explicit path recovery; original failed receipt retained"}], "producers": producers}
    save(root / "provenance.json", manifest)
    return root / "provenance.json", manifest


def edit_receipt(root, manifest, change):
    path = root / "cells/vla4-runtime_update.json"
    receipt = json.loads(path.read_text())
    change(receipt)
    save(path, receipt)
    manifest["producers"][2]["cells"]["vla4-runtime_update"] = digest(path)


def test_accepts_preserved_old_matrices_and_explicit_source_update_without_execution_claim(tmp_path):
    path, manifest = setup_study(tmp_path)
    before = {str(p): digest(p) for p in tmp_path.rglob("*.json")}
    report = audit_provenance(path)
    assert report["status"] == "passed"
    assert report["counts"] == {"passed_cells": 29, "failed_cells": 0, "incomplete_cells": 0}
    assert report["matrix_extension"] == {"original_cells": 28, "main_cells": 24, "operating_point_cells": 4,
                                           "runtime_update_cells": 1, "receipt_relocations": manifest["receipt_relocations"]}
    assert all(row["source_files"] == 4 and row["installed_source_files"] == 3 and row["external_source_files"] == 1 for row in report["cells"])
    assert all(not row["installed_capture_module"]["listed_in_receipt_sources"] for row in report["cells"])
    assert report["producers"][0]["source_inventories"][1]["phase"] == "post_execution"
    assert not report["source_execution_attested"] and not report["complete_module_coverage_attested"]
    assert not report["recommended"] and not report["task_quality_validated"]
    assert before == {str(p): digest(p) for p in tmp_path.rglob("*.json")}


@pytest.mark.parametrize("problem", ["original_hash", "final_hash", "producer_hash", "inventory_hash", "installation_hash", "receipt_hash"])
def test_rejects_changed_bound_bytes(tmp_path, problem):
    path, manifest = setup_study(tmp_path)
    references = {"original_hash": manifest["original_matrix"], "final_hash": manifest["matrix"],
                  "producer_hash": manifest["producers"][0]["matrix"],
                  "inventory_hash": manifest["producers"][0]["source_inventories"][0],
                  "installation_hash": manifest["producers"][0]["installation"]}
    if problem == "receipt_hash":
        manifest["producers"][0]["cells"]["vla4-eager_native"] = "0" * 64
    else:
        references[problem]["sha256"] = "0" * 64
    save(path, manifest)
    assert audit_provenance(path)["status"] == "failed"


@pytest.mark.parametrize("problem", ["old_cell", "old_policy", "main_removed", "extra_removed", "added_operating_point",
                                     "experimental_update", "update_count", "no_relocation", "extra_relocation_field"])
def test_rejects_rewriting_original_matrix_or_inserting_unrelated_candidates(tmp_path, problem):
    path, manifest = setup_study(tmp_path)
    final_path = tmp_path / "matrix_final.json"
    final = json.loads(final_path.read_text())
    if problem == "old_cell":
        final["cells"][0]["revision"] = "b" * 40
    elif problem == "old_policy":
        final["protocol"]["warmup"] = 2
    elif problem == "main_removed":
        final["cells"].pop(0)
    elif problem == "extra_removed":
        final["cells"].pop(24)
    elif problem == "added_operating_point":
        final["cells"][-1]["arm"] = "operating_point"
    elif problem == "experimental_update":
        final["cells"][-1]["experimental"] = True
    elif problem == "update_count":
        final["expected_runtime_update_cells"] = 2
    elif problem == "no_relocation":
        manifest["receipt_relocations"] = []
    else:
        next(cell for cell in final["cells"] if cell["id"] == "edge-eager_native")["revision"] = "b" * 40
    save(final_path, final)
    manifest["matrix"] = reference(tmp_path, final_path.name)
    manifest["producers"][2]["matrix"] = reference(tmp_path, final_path.name)
    save(path, manifest)
    assert audit_provenance(path)["status"] == "failed"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "wrong_producer_declaration"])
def test_requires_one_exact_producer_assignment_for_each_final_cell(tmp_path, problem):
    path, manifest = setup_study(tmp_path)
    assigned = manifest["producers"][0]["cells"]
    if problem == "missing":
        assigned.pop("vla4-eager_native")
    elif problem == "duplicate":
        manifest["producers"][1]["cells"]["vla4-eager_native"] = assigned["vla4-eager_native"]
    elif problem == "unknown":
        assigned["absent"] = "a" * 64
    else:
        manifest["producers"][1]["matrix"] = manifest["original_matrix"]
    save(path, manifest)
    assert audit_provenance(path)["status"] == "failed"


def test_producer_matrix_protocol_must_match_the_frozen_original(tmp_path):
    path, manifest = setup_study(tmp_path)
    producer_matrix = json.loads((tmp_path / "matrix_final.json").read_text())
    producer_matrix["protocol"]["warmup"] = 0
    save(tmp_path / "unmatched_producer.json", producer_matrix)
    manifest["producers"][2]["matrix"] = reference(tmp_path, "unmatched_producer.json")
    edit_receipt(tmp_path, manifest, lambda receipt: receipt.update(matrix_sha256=manifest["producers"][2]["matrix"]["sha256"]))
    save(path, manifest)
    report = audit_provenance(path)
    assert report["status"] == "failed"
    assert report["producers"][2]["errors"] == ["producer matrix protocol changed: protocol"]


def test_second_update_retains_first_updates_frozen_producer_matrix_and_receipt(tmp_path):
    path, manifest = setup_study(tmp_path)
    first_receipt = tmp_path / "cells/vla4-runtime_update.json"
    before = digest(first_receipt)
    final = json.loads((tmp_path / "matrix_final.json").read_text())
    update = copy.deepcopy(next(cell for cell in final["cells"] if cell["id"] == "vla2-runtime_selected"))
    update.update(id="vla2-runtime_update", arm="runtime_update", receipt="cells/vla2-runtime_update.json",
                  baseline_cell="vla2-runtime_selected")
    final["cells"].append(update)
    final["expected_runtime_update_cells"] = 2
    save(tmp_path / "matrix_update_vla2.json", final)
    manifest["matrix"] = reference(tmp_path, "matrix_update_vla2.json")
    producer = copy.deepcopy(manifest["producers"][0])
    producer.update(id="update_vla2", matrix=reference(tmp_path, "matrix_update_vla2.json"))
    receipt = json.loads((tmp_path / "cells/vla2-runtime_selected.json").read_text())
    receipt.update(cell_id=update["id"], arm=update["arm"], matrix_sha256=producer["matrix"]["sha256"])
    save(tmp_path / update["receipt"], receipt)
    producer["cells"] = {update["id"]: digest(tmp_path / update["receipt"])}
    manifest["producers"].append(producer)
    save(path, manifest)
    report = audit_provenance(path)
    assert report["status"] == "passed" and report["counts"]["passed_cells"] == 30
    assert report["matrix_extension"]["runtime_update_cells"] == 2
    assert digest(first_receipt) == before
    first = next(row for row in report["cells"] if row["id"] == "vla4-runtime_update")
    assert first["producer_matrix_sha256"] != manifest["matrix"]["sha256"]


@pytest.mark.parametrize("problem", ["matrix_hash", "empty", "changed_hash", "unknown_path", "relative_path", "parent_path",
                                     "wrong_interpreter", "missing_adapter", "other_family", "not_ok", "wrong_revision"])
def test_sources_and_producer_identity_require_exact_installed_paths_and_hashes(tmp_path, problem):
    path, manifest = setup_study(tmp_path)
    def change(receipt):
        sources = receipt["sources"]
        source = next(iter(sources))
        if problem == "matrix_hash":
            receipt["matrix_sha256"] = manifest["original_matrix"]["sha256"]
        elif problem == "empty":
            receipt["sources"] = {}
        elif problem == "changed_hash":
            sources[source] = "0" * 64
        elif problem == "unknown_path":
            sources["/another/instinctflash/__init__.py"] = sources.pop(source)
        elif problem == "relative_path":
            sources["instinctflash/__init__.py"] = sources.pop(source)
        elif problem == "parent_path":
            sources["/installed/unused/../vla4/model.py"] = sources.pop(source)
        elif problem == "wrong_interpreter":
            receipt["interpreter"] = "/installed/vla2/bin/python"
        elif problem == "missing_adapter":
            sources.pop(next(key for key in sources if "/lingbot_vla_iwm/" in key))
        elif problem == "other_family":
            sources[source.replace("/vla4/", "/vla2/")] = sources.pop(source)
        elif problem == "not_ok":
            receipt["ok"] = False
        else:
            receipt["revision"] = "b" * 40
    edit_receipt(tmp_path, manifest, change)
    save(path, manifest)
    report = audit_provenance(path)
    assert report["status"] == "failed" and report["counts"]["failed_cells"] == 1


@pytest.mark.parametrize("problem", ["missing_capture", "conflicting", "bad_phase", "bad_hash", "bad_path"])
def test_bound_inventory_integrity_and_observation_timing(tmp_path, problem):
    path, manifest = setup_study(tmp_path)
    producer = manifest["producers"][2]
    inv_path = tmp_path / "installed_files_update.json"
    inventory = json.loads(inv_path.read_text())
    if problem == "missing_capture":
        inventory.pop("/installed/vla4/lib/python3.12/site-packages/benchmarks/regression/user_e2e.py")
    elif problem == "conflicting":
        producer["source_inventories"].append(reference(tmp_path, "installed_files.json", phase="pre_execution"))
    elif problem == "bad_phase":
        producer["source_inventories"][1]["phase"] = "frozen"
    elif problem == "bad_hash":
        inventory[next(iter(inventory))] = 12
    else:
        inventory["relative/model.py"] = "a" * 64
    save(inv_path, inventory)
    producer["source_inventories"][0] = reference(tmp_path, inv_path.name, phase="pre_execution")
    save(path, manifest)
    assert audit_provenance(path)["status"] == "failed"


@pytest.mark.parametrize("problem", ["missing_receipt", "unbound_receipt", "missing_inventory", "unbound_inventory"])
def test_future_artifacts_are_incomplete_not_failed_or_passed(tmp_path, problem):
    path, manifest = setup_study(tmp_path)
    if problem == "missing_receipt":
        (tmp_path / "cells/vla4-runtime_update.json").unlink()
    elif problem == "unbound_receipt":
        manifest["producers"][2]["cells"]["vla4-runtime_update"] = None
    elif problem == "missing_inventory":
        (tmp_path / "installed_files_update.json").unlink()
    else:
        manifest["producers"][2]["source_inventories"][0]["sha256"] = None
    save(path, manifest)
    report = audit_provenance(path)
    assert report["status"] == "incomplete"
    assert report["counts"] == {"passed_cells": 28, "failed_cells": 0, "incomplete_cells": 1}


def test_duplicate_json_keys_fail_without_silent_reassignment(tmp_path):
    path, _ = setup_study(tmp_path)
    path.write_text('{"schema":1,"schema":1}')
    report = audit_provenance(path)
    assert report["status"] == "failed" and "duplicate JSON key" in report["errors"][0]


@pytest.mark.parametrize("changed", [False, True])
def test_excluded_failed_receipt_is_hash_bound_without_becoming_a_selected_result(tmp_path, changed):
    path, manifest = setup_study(tmp_path)
    failed = tmp_path / "cells/edge-eager_native.json"
    manifest["bindings"] = [reference(tmp_path, str(failed.relative_to(tmp_path)), reason="Preserve the failed original before path recovery")]
    save(path, manifest)
    if changed:
        save(failed, {"ok": True})
    report = audit_provenance(path)
    assert report["status"] == ("failed" if changed else "passed")
    assert report["counts"]["passed_cells"] == 29
    assert not any(row.get("receipt", {}).get("path") == str(failed) for row in report["cells"])


def test_cli_and_import_need_no_numpy_torch_or_model_environment(tmp_path, capsys):
    path, _ = setup_study(tmp_path)
    output = tmp_path / "report.json"
    assert main(["--manifest", str(path), "--json-output", str(output)]) == 0
    assert json.loads(output.read_text())["counts"]["passed_cells"] == 29
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    import benchmarks.regression.user_provenance as module
    script = "import runpy,sys; runpy.run_path(sys.argv[1],run_name='audit_import'); assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable, "-I", "-c", script, str(Path(module.__file__).resolve())],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
