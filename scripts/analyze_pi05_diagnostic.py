#!/usr/bin/env python3
"""Summarize task-diagnostic traces and draw camera contact sheets on CPU.

Geometric labels are descriptive heuristics, not a replacement for the LIBERO
success predicate or a proof of why a policy failed. No GPU or model is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from benchmarks.vla.pi05_sm120_diagnostic import ARMS, read_json, summarize, validate_trace
from benchmarks.vla.util import sha256_file, write_json_atomic


def geometry(row, arrays):
    goal = row["goal_predicates"]
    if len(goal) != 1 or len(goal[0]) != 3 or goal[0][0].lower() != "on":
        raise ValueError("geometry analysis supports one binary On goal only")
    names = list(arrays["body_names"])
    moving, target = [names.index(name + "_main") for name in goal[0][1:]]
    positions = arrays["body_positions"]
    if positions.shape != (row["steps"] + 1, len(names), 3) or not np.isfinite(positions).all():
        raise ValueError("invalid body position trace")
    object_pos, target_pos = positions[:, moving], positions[:, target]
    xy = np.linalg.norm((object_pos - target_pos)[:, :2], axis=1)
    lift = float((object_pos[:, 2] - object_pos[0, 2]).max())
    near = float(xy.min()) < .07
    if row["success"]:
        label = "success"
    elif lift < .04:
        label = "failed_without_4cm_lift"
    elif near:
        label = "lifted_and_near_target_but_goal_false"
    else:
        label = "lifted_but_never_within_7cm_xy_of_target"
    other_motion = {}
    for index, name in enumerate(names):
        if index in (moving, target) or not name.endswith("_main"):
            continue
        delta = positions[:, index] - positions[0, index]
        distance = float(np.linalg.norm(delta, axis=1).max())
        if distance > .05:
            other_motion[str(name)] = {"max_displacement_m": distance, "max_lift_m": float(delta[:, 2].max())}
    return {"seed": row["seed"], "success": row["success"], "steps": row["steps"],
        "label": label, "max_lift_from_initial_m": lift, "minimum_target_xy_distance_m": float(xy.min()),
        "final_target_xy_distance_m": float(xy[-1]),
        "final_xy_violates_libero_on": bool(xy[-1] >= .03),
        "never_within_libero_on_xy": bool(xy.min() >= .03),
        "final_object_minus_target_z_m": float(object_pos[-1, 2] - target_pos[-1, 2]),
        "other_bodies_moved_over_5cm": other_motion,
        "goal_ever_true": bool(arrays["goal_satisfied"].all(axis=1).any())}


def contact_sheet(arrays, path, title):
    from PIL import Image, ImageDraw
    count = len(arrays["image"])
    indices = np.unique(np.linspace(0, count - 1, min(count, 6), dtype=int))
    canvas = Image.new("RGB", (1024, 40 + ((len(indices) + 1) // 2) * 282), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    for cell, index in enumerate(indices):
        x, y = (cell % 2) * 512, 40 + (cell // 2) * 282
        frame = np.concatenate([arrays["image"][index], arrays["wrist_image"][index]], axis=1)
        canvas.paste(Image.fromarray(frame), (x, y + 20))
        draw.text((x + 5, y + 3), f"Before action step {index * 10} | main / wrist", fill="black")
    canvas.save(path)


def initial_action_comparison(left, right):
    for name in ("image", "wrist_image", "input_state", "noises"):
        if not np.array_equal(left[name][0], right[name][0]):
            raise ValueError("initial numeric comparison does not have identical inputs/noise")
    a, b = left["chunks"][0].astype(float).ravel(), right["chunks"][0].astype(float).ravel()
    return {"cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "max_abs_delta": float(np.abs(a - b).max()), "mean_abs_delta": float(np.abs(a - b).mean())}


def portable_evidence(result, analysis):
    """Keep paired outcomes and hashes, not local paths, images or simulator dumps."""
    pairs = []
    for index, seed in enumerate(result["plan"]["seeds"]):
        rows = {arm: result["episodes"][arm][index] for arm in ARMS}
        pairs.append({"seed": seed, "initial_observation_sha256": rows["official"]["scene"]["initial_observation_sha256"],
            "arms": {arm: {k: row[k] for k in ("success", "steps", "action_digest", "noise_sha256", "artifacts")}
                     for arm, row in rows.items()}})
    kept = ("protocol", "status", "task", "prompt", "episodes_per_arm", "arm_order", "successes", "outcome_groups",
            "comparisons", "historical_replay_mismatches", "historical_scale_changes", "matched_noise_calls",
            "official_loaded_weights", "source_sha256", "environment", "receipt_sha256", "limitations")
    return {**{k: result[k] for k in kept}, "schema_version": 1, "pairs": pairs,
            "baseline_sha256": result["plan"]["baseline_sha256"],
            "checkpoint_config_sha256": result["plan"]["checkpoint_config_sha256"],
            "analysis": {k: analysis[k] for k in ("episodes", "labels", "initial_action_vs_official",
                        "analysis_source_sha256", "thresholds", "limitations")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="Inspect completed episodes while a campaign is running")
    parser.add_argument("--sheet-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--evidence", type=Path, help="Export compact, source-verified evidence after a complete run")
    args = parser.parse_args()
    if args.evidence and args.partial:
        parser.error("partial results cannot be exported as complete evidence")
    if not args.partial and not (args.campaign / "results.json").exists():
        parser.error("campaign is incomplete; use --partial for exploratory inspection")
    args.output.mkdir(parents=True, exist_ok=True)
    episodes, files, images = {}, {}, {}
    for arm in ARMS:
        receipt = args.campaign / f"{arm}.json"
        if not receipt.exists():
            if args.partial:
                continue
            raise ValueError("missing arm receipt")
        files[receipt.name] = sha256_file(receipt)
        episodes[arm] = []
        for row in read_json(receipt)["episodes"]:
            name = f"{arm}/{row['seed']}.npz"
            path = args.campaign / name
            if sha256_file(path) != row["artifacts"][name]:
                raise ValueError("trace changed before analysis")
            with np.load(path, allow_pickle=False) as arrays:
                validate_trace(row, arrays)
                episodes[arm].append(geometry(row, arrays))
                if row["seed"] in args.sheet_seeds:
                    image = args.output / f"{arm}-{row['seed']}.png"
                    contact_sheet(arrays, image, f"{arm}, seed {row['seed']}, success={row['success']}, steps={row['steps']}")
                    images[image.name] = sha256_file(image)
    first_action = {}
    if all(a in episodes for a in ARMS):
        for arm in ("native", "fp8"):
            comparisons = []
            common = set(r["seed"] for r in episodes["official"]) & set(r["seed"] for r in episodes[arm])
            for seed in sorted(common):
                with np.load(args.campaign / f"official/{seed}.npz") as a, np.load(args.campaign / f"{arm}/{seed}.npz") as b:
                    comparisons.append({"seed": seed, **initial_action_comparison(a, b)})
            first_action[arm] = comparisons
    summary = {"status": "PARTIAL_DIAGNOSTIC" if args.partial else "DIAGNOSTIC", "episodes": episodes,
        "labels": {a: dict(Counter(r["label"] for r in rows)) for a, rows in episodes.items()},
        "initial_action_vs_official": first_action, "receipt_sha256": files, "contact_sheet_sha256": images,
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "thresholds": {"lift_m": .04, "target_xy_m": .07, "pinned_libero_on_xy_m": .03,
                       "other_body_displacement_m": .05},
        "limitations": ["Post-hoc geometric description; not a causal classification of failure.",
                        "4 cm lift and 7 cm proximity are analysis thresholds, not LIBERO success thresholds.",
                        "Contact sheets show policy observation frames, not every simulator step; MP4 files contain every step."]}
    write_json_atomic(args.output / "analysis.json", summary)
    if args.evidence:
        result = summarize(args.campaign.resolve())
        if len(result["plan"]["seeds"]) != 50:
            raise ValueError("portable task evidence requires all 50 planned seeds")
        if result["receipt_sha256"] != {a: files[f"{a}.json"] for a in ARMS}:
            raise ValueError("receipts changed between analysis and export")
        write_json_atomic(args.evidence, portable_evidence(result, summary))
    print(json.dumps(summary["labels"], indent=2))


if __name__ == "__main__":
    main()
