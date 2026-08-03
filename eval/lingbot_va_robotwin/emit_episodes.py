#!/usr/bin/env python3
"""Turn a RoboTwin run's save_root into per-episode JSONL for certification.

`aggregate.py` produces the reportable macro number. Certification needs something aggregates
cannot give: a PER-EPISODE outcome keyed by (task, episode index, seed), so two runs can be paired
episode-for-episode. Two independent 2500-episode aggregates would let ordinary between-run
variance masquerade as a real difference.

The evidence is already on disk and needs no harness change. The client writes
`stseed-<S>/visualization/<task>/<n>_<prompt>_<True|False>.mp4`, which carries all four fields:
the seed is the directory, the task is the parent, and the index and outcome are the filename.

Cross-checked against `stseed-<S>/metrics/<task>/res.json` exactly as `aggregate.py` does, and it
REFUSES rather than emitting a partial file if the two sources disagree — the same discipline as
`REPORTABLE: NO`. A certificate built on a run that silently lost episodes is worse than no
certificate.

    python emit_episodes.py <save_root> -o teacher.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MP4_RE = re.compile(r"^(\d+)_(.*)_(True|False)\.mp4$")
STSEED_RE = re.compile(r"^stseed-(\d+)$")


def collect(save_root: Path):
    """(episodes, problems). One record per mp4; problems block emission."""
    episodes, problems = [], []
    seen: set[tuple[str, int, int]] = set()
    per_task_mp4: dict[str, list[int]] = {}

    roots = sorted(save_root.glob("stseed-*"))
    if not roots:
        problems.append(f"no stseed-* directories under {save_root}")

    for sroot in roots:
        m = STSEED_RE.match(sroot.name)
        if not m:
            problems.append(f"cannot parse a seed from {sroot.name!r}")
            continue
        st_seed = int(m.group(1))
        for vis_task in sorted((sroot / "visualization").glob("*")):
            if not vis_task.is_dir():
                continue
            task = vis_task.name
            for mp4 in sorted(vis_task.glob("*.mp4")):
                mm = MP4_RE.match(mp4.name)
                if not mm:
                    problems.append(f"unparseable episode file {mp4}")
                    continue
                idx, success = int(mm.group(1)), mm.group(3) == "True"
                key = (task, idx, st_seed)
                if key in seen:
                    # the failure mode aggregate.py documents: two clients writing one save_root
                    problems.append(f"duplicate episode {key}; two clients may share a save_root")
                    continue
                seen.add(key)
                per_task_mp4.setdefault(task, []).append(idx)
                episodes.append({
                    "episode_id": f"{task}/{idx}",
                    "seed": st_seed,
                    "task": task,
                    "success": success,
                })

    # independent second source, same cross-check aggregate.py performs
    for res in sorted(save_root.glob("stseed-*/metrics/*/res.json")):
        task = res.parent.name
        try:
            d = json.loads(res.read_text())
        except Exception as e:
            problems.append(f"{res} unreadable ({e})")
            continue
        jt = d.get("total_num")
        got = len(per_task_mp4.get(task, []))
        if jt is not None and int(jt) != got:
            problems.append(f"{task}: {got} episode files but res.json says total_num={int(jt)}")
    return episodes, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save_root")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--allow-problems", action="store_true",
                    help="emit anyway. Only for debugging; a certificate built on this is void.")
    a = ap.parse_args()

    episodes, problems = collect(Path(a.save_root))
    for p in problems:
        print(f"  PROBLEM: {p}", file=sys.stderr)
    if problems and not a.allow_problems:
        print(f"\nEMITTED: NO -- {len(problems)} problem(s). Certification refuses a run whose "
              f"episode evidence is inconsistent; fix the run rather than the threshold.",
              file=sys.stderr)
        return 1

    with open(a.out, "w") as f:
        for e in sorted(episodes, key=lambda x: (x["task"], x["seed"], x["episode_id"])):
            f.write(json.dumps(e) + "\n")
    n_ok = sum(1 for e in episodes if e["success"])
    print(f"wrote {len(episodes)} episodes to {a.out}  "
          f"({n_ok} success, {len(episodes) - n_ok} failure, "
          f"{n_ok / max(len(episodes), 1):.3f})")
    if problems:
        print("WARNING: emitted with problems because --allow-problems was passed. "
              "Any certificate from this file is void.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
