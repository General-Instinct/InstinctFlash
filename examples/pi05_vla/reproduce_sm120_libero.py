#!/usr/bin/env python3
"""Run the source-locked FlashRT matched LIBERO campaign through the public driver."""
import argparse
import json
from pathlib import Path

from benchmarks.vla.pi05_sm120_libero import campaign, prepare_calibration, source_hashes, environment_identity
from benchmarks.vla.util import sha256_file, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path,
                        help="Reproduce a published, frozen calibration choice without reselecting it")
    args = parser.parse_args()
    config = {k: str(getattr(args, k).resolve()) for k in ("checkpoint", "dataset", "output")}
    path = args.output.resolve().with_name(args.output.name + "-config.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("existing campaign config differs")
    path.write_text(json.dumps(config, indent=2) + "\n")
    if args.evidence:
        evidence = json.loads(args.evidence.read_text())
        if evidence["source_sha256"] != source_hashes():
            raise ValueError("published evidence belongs to different sources")
        if evidence["environment"] != environment_identity():
            raise ValueError("reproduction environment differs from the published evidence")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        plan_path = output / "plan.json"
        if not plan_path.exists():
            prepare_calibration(config, output)
            if sha256_file(output / "calibration.json") != evidence["protocol"]["calibration_sha256"]:
                raise ValueError("reproduction calibration differs from the published evidence")
            plan = {**evidence["protocol"], "config": config, "source_sha256": source_hashes(),
                    "environment": evidence["environment"],
                    "reproduction_evidence_sha256": sha256_file(args.evidence)}
            write_json_atomic(plan_path, plan)
        elif json.loads(plan_path.read_text()).get("reproduction_evidence_sha256") != sha256_file(args.evidence):
            raise ValueError("resume requested a different evidence artifact")
    return campaign(path)


if __name__ == "__main__":
    raise SystemExit(main())
