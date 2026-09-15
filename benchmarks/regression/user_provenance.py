"""CPU-only provenance consistency audit for an extended public-user study.

Run ``python -m benchmarks.regression.user_provenance --manifest provenance.json
--json-output provenance_report.json`` alongside ``user_report``. The sidecar
manifest has schema=1, hash-bound ``original_matrix`` and ``matrix`` references,
optional ``receipt_relocations`` ({id, from, to, reason}), and ``producers``.
Optional ``bindings`` retain other hash-bound JSON evidence with a ``reason``;
these artifacts are never assigned as selected producer receipts.
Each producer has {id, matrix, installation, source_inventories, cells}; ``cells``
maps assigned cell IDs to receipt SHA-256 values. File references have {path,
sha256}; inventory references also require phase=pre_execution|post_execution.
An inventory is an independently recorded {absolute_installed_path: sha256}
mapping, never a map reconstructed from the receipt hashes it will validate.
Null hashes explicitly leave future artifacts unbound and the audit incomplete.

This checks immutable declarations, receipt bytes, producer matrix selection,
and recorded paths/hashes against bound installation inventories. It does not
attest which code executed, complete module coverage, or task quality. In
particular, post-execution inventories are later observations, and the capture
module runs as __main__, so existing receipts omit its own source entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


MAIN_ARMS = ("eager_native", "runtime_default", "runtime_selected")
ADAPTERS = {"pi05": "pi05_iwm", "vla4": "lingbot_vla_iwm", "vla2": "lingbot_vla_v2_iwm",
            "groot": "groot_n17_iwm", "edge": "cosmos3_iwm", "nano": "cosmos3_iwm",
            "dreamzero": "dreamzero_iwm"}
OWNED_PACKAGES = frozenset(("instinctflash", "flash_rt", "benchmarks", *ADAPTERS.values()))


class InvalidProvenance(ValueError):
    """Existing evidence conflicts with its declared provenance."""


class PendingProvenance(Exception):
    """A future artifact has not been hash-bound yet."""


def _check(condition, message):
    if not condition:
        raise InvalidProvenance(message)


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _check(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise InvalidProvenance(f"nonfinite JSON value: {value}")

    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=unique, parse_constant=invalid)
    _canonical(result)
    return result


def _hash(value, name):
    _check(isinstance(value, str) and len(value) == 64
           and all(char in "0123456789abcdef" for char in value), f"{name} must be a lowercase SHA-256")
    return value


def _path(root, value):
    _check(isinstance(value, str) and bool(value), "artifact path must be nonempty")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _installed_path(value):
    _check(isinstance(value, str) and value.startswith("/") and "\\" not in value,
           "installed path must be an absolute POSIX path")
    path = PurePosixPath(value)
    _check(str(path) == value and ".." not in path.parts and not value.startswith("//"),
           f"installed path must be normalized: {value}")
    return path


def _bound_json(root, reference, artifacts):
    _check(isinstance(reference, dict) and "sha256" in reference, "artifact reference requires path and sha256")
    path = _path(root, reference.get("path"))
    expected = reference["sha256"]
    if expected is None:
        raise PendingProvenance(f"unbound artifact: {path}")
    _hash(expected, "artifact sha256")
    actual = _sha(path)
    _check(actual == expected, f"artifact SHA-256 mismatch: {path}")
    artifacts.append({"path": str(path.resolve()), "sha256": actual})
    return _json(path)


def _cells(matrix):
    _check(isinstance(matrix, dict) and matrix.get("schema") == 1, "matrix schema must be 1")
    cells = matrix.get("cells")
    _check(isinstance(cells, list) and bool(cells), "matrix cells must be a nonempty list")
    result = {}
    for cell in cells:
        _check(isinstance(cell, dict), "matrix cell must be an object")
        identifier = cell.get("id")
        _check(isinstance(identifier, str) and bool(identifier) and identifier not in result,
               "cell IDs must be unique nonempty strings")
        result[identifier] = cell
    return result


def _extension(original, final, relocations):
    old, new = _cells(original), _cells(final)
    _check(isinstance(relocations, list), "receipt_relocations must be a list")
    moves = {}
    for move in relocations:
        _check(isinstance(move, dict) and move.get("id") in old and move["id"] not in moves,
               "receipt relocation must name one unique original cell")
        _check(isinstance(move.get("reason"), str) and bool(move["reason"].strip()),
               "receipt relocation requires a reason")
        identifier = move["id"]
        _check(move.get("from") == old[identifier]["receipt"] and move.get("to") != move["from"],
               "receipt relocation must bind its original and changed paths")
        _check(isinstance(move.get("to"), str) and bool(move["to"]), "receipt relocation requires a target path")
        moves[identifier] = move
    for identifier, cell in old.items():
        _check(cell.get("arm") in (*MAIN_ARMS, "operating_point"), "original matrix may contain only initial arms")
        expected = dict(cell)
        if identifier in moves:
            expected["receipt"] = moves[identifier]["to"]
        _check(identifier in new and _canonical(new[identifier]) == _canonical(expected),
               f"original cell declaration changed: {identifier}")
    for key, value in original.items():
        if key != "cells":
            _check(key in final and _canonical(final[key]) == _canonical(value),
                   f"original matrix protocol changed: {key}")
    additions = [cell for identifier, cell in new.items() if identifier not in old]
    expected_updates = final.get("expected_runtime_update_cells")
    _check(type(expected_updates) is int and expected_updates > 0 and len(additions) == expected_updates,
           "runtime update count differs from expected_runtime_update_cells")
    for cell in additions:
        _check(cell.get("arm") == "runtime_update" and cell.get("experimental") is False,
               "added cells must be nonexperimental runtime_update candidates")
        baseline = old.get(cell.get("baseline_cell"))
        _check(baseline is not None and baseline.get("arm") in ("eager_native", "runtime_selected"),
               "runtime update requires an original eager_native/runtime_selected baseline")
        _check(all(cell.get(key) == baseline.get(key) for key in ("family", "comparison_group")),
               "runtime update baseline must share family and comparison_group")
        _check(cell.get("cross_policy_baseline") is None and cell.get("comparison_baseline") is not True,
               "runtime update cannot promote itself or request a cross-policy comparison")
    main = [cell for cell in old.values() if cell["arm"] in MAIN_ARMS]
    expected_main = final.get("expected_main_cells")
    _check(type(expected_main) is int and len(main) == expected_main, "original main cell count changed")
    families = final.get("expected_families")
    _check(isinstance(families, list) and families and all(isinstance(family, str) for family in families)
           and len(set(families)) == len(families), "expected_families must be unique strings")
    _check({cell.get("family") for cell in main} == set(families), "original family coverage changed")
    for family in families:
        _check(sorted(cell["arm"] for cell in main if cell["family"] == family) == sorted(MAIN_ARMS),
               f"original main arms incomplete: {family}")
    extras = sum(cell["arm"] == "operating_point" for cell in old.values())
    expected_extras = final.get("expected_operating_point_cells")
    _check(type(expected_extras) is int and extras == expected_extras, "original operating-point cell count changed")
    return new, {"original_cells": len(old), "main_cells": len(main), "operating_point_cells": extras,
                 "runtime_update_cells": len(additions), "receipt_relocations": relocations}


def _inventories(root, references, artifacts):
    _check(isinstance(references, list) and bool(references), "producer requires source_inventories")
    merged, observations = {}, []
    for reference in references:
        _check(isinstance(reference, dict) and reference.get("phase") in ("pre_execution", "post_execution"),
               "source inventory must declare pre_execution or post_execution phase")
        inventory = _bound_json(root, reference, artifacts)
        _check(isinstance(inventory, dict) and bool(inventory), "source inventory must be a nonempty path/hash mapping")
        for path, digest in inventory.items():
            _installed_path(path)
            _hash(digest, "installed source sha256")
            _check(path not in merged or merged[path] == digest, f"conflicting source inventories: {path}")
            merged[path] = digest
        observations.append({"path": reference["path"], "sha256": reference["sha256"],
                             "phase": reference["phase"], "files": len(inventory)})
    return merged, observations


def _receipt(root, cell, expected_hash, matrix_hash, installation, inventory):
    path = _path(root, cell["receipt"])
    if expected_hash is None:
        raise PendingProvenance(f"unbound receipt: {path}")
    _hash(expected_hash, "receipt sha256")
    actual = _sha(path)
    _check(actual == expected_hash, f"receipt SHA-256 mismatch: {cell['id']}")
    receipt = _json(path)
    _check(isinstance(receipt, dict) and receipt.get("schema") == 1 and receipt.get("ok") is True,
           "selected receipt must have schema=1 and ok=true")
    for receipt_key, cell_key in (("cell_id", "id"), ("family", "family"), ("arm", "arm"),
                                  ("model_id", "model_id"), ("revision", "revision")):
        _check(receipt.get(receipt_key) == cell.get(cell_key), f"receipt differs from declaration: {receipt_key}")
    _check(_hash(receipt.get("matrix_sha256"), "producer matrix_sha256") == matrix_hash,
           "receipt producer matrix_sha256 mismatch")
    _check(isinstance(installation, dict) and installation.get("schema") == 1 and installation.get("ok") is True,
           "installation must have schema=1 and ok=true")
    environment = installation["environments"][cell["family"]]
    interpreter = str(_installed_path(environment["python"]))
    site = _installed_path(environment["overlay_site"])
    _check(receipt.get("interpreter") == interpreter, "receipt interpreter differs from bound installation")
    sources = receipt.get("sources")
    _check(isinstance(sources, dict) and bool(sources), "receipt sources must be a nonempty path/hash mapping")
    installed_count = 0
    for name, digest in sources.items():
        source = _installed_path(name)
        _check(source.suffix in (".py", ".so"), "receipt source must be a Python file or shared library")
        _hash(digest, "receipt source sha256")
        _check(name in inventory, f"receipt source absent from bound inventories: {name}")
        _check(inventory[name] == digest, f"receipt source SHA-256 mismatch: {name}")
        inside = source.is_relative_to(site)
        if OWNED_PACKAGES.intersection(source.parts):
            _check(inside, f"owned package source outside family installation: {name}")
        installed_count += int(inside)
    required = ("instinctflash/__init__.py", "instinctflash/runtime/facade.py")
    adapter = ADAPTERS.get(cell["family"])
    required += ((f"{adapter}/adapter.py",) if adapter else ("instinctflash/adapters/lingbot_va.py",))
    for relative in required:
        _check(str(site / relative) in sources, f"receipt missing installed source: {relative}")
    producer_module = str(site / "benchmarks/regression/user_e2e.py")
    _check(producer_module in inventory, "installed capture module absent from bound inventories")
    return {"receipt": {"path": str(path.resolve()), "sha256": actual},
            "producer_matrix_sha256": matrix_hash, "interpreter": interpreter,
            "source_files": len(sources), "installed_source_files": installed_count,
            "external_source_files": len(sources) - installed_count,
            "installed_capture_module": {"path": producer_module, "sha256": inventory[producer_module],
                                         "listed_in_receipt_sources": producer_module in sources}}


def audit_provenance(manifest_path, root=None):
    """Validate an opt-in sidecar without loading any model or remote path."""
    manifest_path = Path(manifest_path).resolve()
    root = Path(root).resolve() if root is not None else manifest_path.parent
    report = {"schema": "instinctflash.user_provenance.v1", "status": "passed", "errors": [], "pending": [],
              "scope": "Recorded provenance consistency; inventory timing is explicit; no source execution or quality attestation",
              "source_execution_attested": False, "complete_module_coverage_attested": False,
              "task_quality_validated": False, "recommended": False, "artifacts": [], "producers": [], "cells": []}
    try:
        report["manifest"] = {"path": str(manifest_path), "sha256": _sha(manifest_path)}
        manifest = _json(manifest_path)
        _check(isinstance(manifest, dict) and manifest.get("schema") == 1, "provenance manifest schema must be 1")
        original = _bound_json(root, manifest.get("original_matrix"), report["artifacts"])
        final = _bound_json(root, manifest.get("matrix"), report["artifacts"])
        cells, report["matrix_extension"] = _extension(original, final, manifest.get("receipt_relocations", []))
        bindings = manifest.get("bindings", [])
        _check(isinstance(bindings, list), "bindings must be a list")
        report["retained_bindings"] = []
        for binding in bindings:
            _check(isinstance(binding, dict) and isinstance(binding.get("reason"), str)
                   and bool(binding["reason"].strip()), "retained binding requires a reason")
            try:
                _bound_json(root, binding, report["artifacts"])
                report["retained_bindings"].append({**binding, "status": "passed"})
            except (FileNotFoundError, PendingProvenance) as error:
                report["pending"].append(str(error))
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
                report["errors"].append(str(error))
        producers = manifest.get("producers")
        _check(isinstance(producers, list) and bool(producers), "producers must be a nonempty list")
        assigned, names = set(), set()
        for producer in producers:
            _check(isinstance(producer, dict), "producer must be an object")
            name = producer.get("id")
            _check(isinstance(name, str) and bool(name) and name not in names, "producer IDs must be unique strings")
            names.add(name)
            assignments = producer.get("cells")
            _check(isinstance(assignments, dict) and bool(assignments), "producer cells must map IDs to receipt hashes")
            _check(not assigned.intersection(assignments) and set(assignments).issubset(cells),
                   "producer cell assignments must be unique and declared in the final matrix")
            assigned.update(assignments)
        _check(assigned == set(cells), "every final cell requires exactly one producer assignment")
        for producer in producers:
            entry = {"id": producer["id"], "status": "passed", "cells": list(producer["cells"])}
            try:
                producer_matrix = _bound_json(root, producer.get("matrix"), report["artifacts"])
                declarations = _cells(producer_matrix)
                for key, value in original.items():
                    if key != "cells":
                        _check(key in producer_matrix and _canonical(producer_matrix[key]) == _canonical(value),
                               f"producer matrix protocol changed: {key}")
                for identifier in producer["cells"]:
                    _check(identifier in declarations and _canonical(cells[identifier]) == _canonical(declarations[identifier]),
                           f"assigned producer cell declaration differs from final matrix: {identifier}")
                installation = _bound_json(root, producer.get("installation"), report["artifacts"])
                inventory, entry["source_inventories"] = _inventories(root, producer.get("source_inventories"), report["artifacts"])
                for identifier, expected in producer["cells"].items():
                    row = {"id": identifier, "producer": producer["id"], "status": "passed"}
                    try:
                        row.update(_receipt(root, cells[identifier], expected, producer["matrix"]["sha256"], installation, inventory))
                    except (FileNotFoundError, PendingProvenance) as error:
                        row.update(status="incomplete", pending=[str(error)])
                    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
                        row.update(status="failed", errors=[str(error)])
                    report["cells"].append(row)
            except (FileNotFoundError, PendingProvenance) as error:
                entry.update(status="incomplete", pending=[str(error)])
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
                entry.update(status="failed", errors=[str(error)])
            if entry["status"] != "passed":
                detail = "pending" if entry["status"] == "incomplete" else "errors"
                for identifier in producer["cells"]:
                    report["cells"].append({"id": identifier, "producer": producer["id"],
                                            "status": entry["status"], detail: list(entry[detail])})
            report["producers"].append(entry)
    except (FileNotFoundError, PendingProvenance) as error:
        report["pending"].append(str(error))
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        report["errors"].append(str(error))
    statuses = [row["status"] for row in report["cells"] + report["producers"]]
    if report["errors"] or "failed" in statuses:
        report["status"] = "failed"
    elif report["pending"] or "incomplete" in statuses:
        report["status"] = "incomplete"
    report["counts"] = {"passed_cells": sum(row["status"] == "passed" for row in report["cells"]),
                        "failed_cells": sum(row["status"] == "failed" for row in report["cells"]),
                        "incomplete_cells": sum(row["status"] == "incomplete" for row in report["cells"])}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_provenance(args.manifest, args.root)
    args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["counts"]}, sort_keys=True))
    return {"passed": 0, "failed": 1, "incomplete": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
