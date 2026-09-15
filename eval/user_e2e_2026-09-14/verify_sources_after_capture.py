"""Read the actual installed/vendor files after capture and compare producer hashes.

The receipt supplies the path inventory, never the expected output hashes. This
is a post-execution consistency observation, not a pre-execution vendor freeze.
"""
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    inventory_path = ROOT / "vendor_sources_post_execution.json"
    report_path = ROOT / "source_post_execution_check.json"
    assert not inventory_path.exists() and not report_path.exists()
    matrix = json.loads((ROOT / "matrix_update_vla2.json").read_text())
    observed, vendor, receipts, errors = {}, {}, {}, []
    owned_root = ROOT / "envs"
    for cell in matrix["cells"]:
        path = ROOT / cell["receipt"]
        receipt = json.loads(path.read_text())
        assert receipt["ok"], path
        receipts[cell["id"]] = sha(path)
        for filename, expected in receipt["sources"].items():
            source = Path(filename)
            if filename not in observed:
                observed[filename] = sha(source)
            if observed[filename] != expected:
                errors.append(dict(cell=cell["id"], path=filename,
                                   recorded_sha256=expected, observed_sha256=observed[filename]))
            if not source.is_relative_to(owned_root):
                vendor[filename] = observed[filename]
    inventory_path.write_text(json.dumps(vendor, indent=2, sort_keys=True)+"\n")
    report = dict(schema=1, ok=not errors, phase="post_execution",
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        collector_sha256=sha(__file__), matrix_sha256=sha(ROOT / "matrix_update_vla2.json"),
        receipt_sha256=receipts, checked_source_paths=len(observed), vendor_source_paths=len(vendor),
        vendor_inventory_sha256=sha(inventory_path), errors=errors,
        scope="Actual bytes read after capture; original installed inventories remain separate; no claim of complete module or dependency coverage")
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("ok", "checked_source_paths", "vendor_source_paths")}))
    assert not errors, errors


if __name__ == "__main__":
    main()
