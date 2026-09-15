"""Read-only source-manifest delta; write a new local audit receipt only."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--correspondence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.before / "manifest.json").read_text())
    after = json.loads((args.after / "manifest.json").read_text())
    correspondence = json.loads(args.correspondence.read_text())
    assert correspondence["status"] == "passed_source_and_plan_correspondence"
    assert correspondence["refs"]["stage_manifest"]["sha256"] == digest(
        args.after / "manifest.json"
    )
    old, new = before["files"], after["files"]
    changed = sorted(k for k in old.keys() & new.keys() if old[k]["sha256"] != new[k]["sha256"])
    added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
    for stage, manifest in ((args.before, old), (args.after, new)):
        for name in changed + added + removed:
            if name in manifest:
                assert digest(stage / "source" / name) == manifest[name]["sha256"], name
    protected = (
        "instinctflash/", "serving/flash_rt/", "serving/csrc/", "serving/native/",
        "serving/third_party/", "examples/cosmos3_policy/cosmos3_iwm/",
        "examples/pi05_vla/pi05_iwm/", "examples/groot_n17/groot_n17_iwm/",
        "examples/lingbot_vla/lingbot_vla_iwm/", "examples/lingbot_vla_v2/lingbot_vla_v2_iwm/",
        "examples/dreamzero/dreamzero_iwm/",
    )
    assert not any(name.startswith(protected) for name in changed + added + removed)
    result = {
        "kind": "public_stage_source_delta_v1",
        "status": "passed_no_internal_inference_source_drift",
        "auditor_sha256": digest(Path(__file__)),
        "before": {"path": str(args.before), "manifest_sha256": digest(args.before / "manifest.json")},
        "after": {"path": str(args.after), "manifest_sha256": digest(args.after / "manifest.json")},
        "correspondence": {"path": str(args.correspondence), "sha256": digest(args.correspondence)},
        "changed": {name: {"before": old[name]["sha256"], "after": new[name]["sha256"]} for name in changed},
        "added": {name: new[name]["sha256"] for name in added},
        "removed": {name: old[name]["sha256"] for name in removed},
        "unchanged_files": len(old.keys() & new.keys()) - len(changed),
        "scope": "Manifest byte comparison plus changed-file rehash; earlier correspondence separately verifies inference and native payloads.",
        "limitations": [
            "Dependency metadata and preparation helper changes require their separate install/functional evidence.",
            "No new inference, numerical, task-quality or public-download availability claim.",
        ],
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "changed": len(changed), "added": len(added), "removed": len(removed), "receipt_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
