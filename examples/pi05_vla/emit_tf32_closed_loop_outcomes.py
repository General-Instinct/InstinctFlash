#!/usr/bin/env python3
"""Normalize one LIBERO arm's evaluator JSONL into the preregistered certificate schema.

Raw input is the runner's identity-rich JSONL. The emitter refuses any task, seed, initial-state,
policy-seed, checkpoint or arm mismatch before deriving a stable ``episode_id``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "tf32_closed_loop_preregistration.json"


def emit(input_path: Path, output_path: Path, *, arm: str, allow_partial: bool = False) -> int:
    declaration = json.loads(PREREG.read_text())
    rows = []
    seen = set()
    indexes: dict[str, set[int]] = {}
    seed_base = int(declaration["seed_base"])
    seed_stride = int(declaration["seed_stride"])
    expected_each = int(declaration["episodes_per_task"])
    expected_revision = str(declaration["checkpoint"]["revision"])
    expected_task_ids = {int(task_id) for task_id in declaration["task_ids"]}
    with input_path.open() as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                run_id = raw["run_id"]
                raw_arm = raw["arm"]
                suite = raw["suite"]
                task_id = raw["task_id"]
                task = raw["task"]
                index = raw["episode_index"]
                seed = raw["seed"]
                init_state_id = raw["init_state_id"]
                policy_seed = raw["policy_seed"]
                checkpoint_revision = raw["checkpoint_revision"]
                success = raw["success"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_no}: invalid raw outcome ({exc})") from exc
            integer_fields = {
                "task_id": task_id, "episode_index": index, "seed": seed,
                "init_state_id": init_state_id, "policy_seed": policy_seed,
            }
            wrong_types = [
                name for name, value in integer_fields.items()
                if not isinstance(value, int) or isinstance(value, bool)
            ]
            if wrong_types:
                raise ValueError(
                    f"{input_path}:{line_no}: integer fields have wrong types: {wrong_types}"
                )
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"{input_path}:{line_no}: run_id must be a non-empty string")
            if not isinstance(success, bool):
                raise ValueError(f"{input_path}:{line_no}: success must be JSON true/false")
            if raw_arm != arm:
                raise ValueError(
                    f"{input_path}:{line_no}: row arm {raw_arm!r} does not match {arm!r}"
                )
            if suite != "libero_spatial":
                raise ValueError(f"{input_path}:{line_no}: suite must be 'libero_spatial'")
            if task_id not in expected_task_ids or task != f"task_{task_id:02d}":
                raise ValueError(
                    f"{input_path}:{line_no}: task identity {(task_id, task)!r} is not preregistered"
                )
            if index < 0 or index >= expected_each:
                raise ValueError(
                    f"{input_path}:{line_no}: episode_index {index} is outside 0..{expected_each - 1}"
                )
            expected_seed = seed_base + seed_stride * task_id + index
            if seed != expected_seed:
                raise ValueError(
                    f"{input_path}:{line_no}: seed {seed} does not match preregistered "
                    f"seed_base+stride*task_id+episode_index={expected_seed}"
                )
            if init_state_id != index or policy_seed != expected_seed:
                raise ValueError(
                    f"{input_path}:{line_no}: init_state_id/policy_seed do not match episode schedule"
                )
            if checkpoint_revision != expected_revision:
                raise ValueError(
                    f"{input_path}:{line_no}: checkpoint revision {checkpoint_revision!r} "
                    f"does not match {expected_revision!r}"
                )
            episode_id = f"libero_spatial/{task}/{index:03d}"
            key = (episode_id, seed)
            if key in seen:
                raise ValueError(f"{input_path}:{line_no}: duplicate {key}")
            seen.add(key)
            indexes.setdefault(task, set()).add(index)
            row = {
                "run_id": run_id, "arm": arm, "suite": suite,
                "task_id": task_id, "task": task, "episode_index": index,
                "episode_id": episode_id, "seed": seed, "init_state_id": init_state_id,
                "policy_seed": policy_seed, "checkpoint_revision": checkpoint_revision,
                "success": success,
            }
            rows.append(row)
    counts = Counter(row["task"] for row in rows)
    if not allow_partial:
        expected_tasks = int(declaration["tasks"])
        expected_names = {f"task_{int(task_id):02d}" for task_id in declaration["task_ids"]}
        if len(counts) != expected_tasks or any(n != expected_each for n in counts.values()):
            raise ValueError(
                f"complete arm requires {expected_tasks} tasks x {expected_each} episodes; "
                f"observed task counts {dict(sorted(counts.items()))}"
            )
        if set(counts) != expected_names:
            raise ValueError(
                f"complete arm requires tasks {sorted(expected_names)}; observed {sorted(counts)}"
            )
        expected_indexes = set(range(expected_each))
        wrong = {task: sorted(ids) for task, ids in indexes.items() if ids != expected_indexes}
        if wrong:
            raise ValueError(f"complete arm has wrong episode indexes: {wrong}")
        if len(rows) != int(declaration["exact_pairs"]):
            raise ValueError(
                f"complete arm requires exactly {declaration['exact_pairs']} rows; got {len(rows)}"
            )
    output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("control_fp32", "treatment_tf32"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        count = emit(args.input, args.output, arm=args.arm, allow_partial=args.allow_partial)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"wrote {count} {args.arm} outcomes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
