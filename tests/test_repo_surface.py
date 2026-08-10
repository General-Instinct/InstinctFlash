#!/usr/bin/env python3
"""The repository root is part of the product. These are the checks that keep it that way.

Written after the root had accumulated 24 markdown files -- LAYER5_QKV_FEASIBILITY.md,
SALVAGE_PR2.md, three overlapping audit records -- which made a maintained framework read as an
internal research notebook. The research is not deleted; it lives in docs/ and in git history. This
test exists so the root does not silently refill.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED: list[str] = []

#: What a visitor should find at the top level, and nothing more.
ALLOWED_ROOT_MD = {"README.md", "ARCHITECTURE.md", "CHECKPOINTS.md"}

#: The four questions the first screen has to answer, in order.
FIRST_SCREEN = ("## Install", "## Load a model", "## Get actions")


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def tracked(pattern: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return [p for p in out if "/" not in p]


def test_root_is_product_shaped():
    print("\n=== 1. the root reads as a framework, not a notebook ===")
    md = set(tracked("*.md"))
    extra = sorted(md - ALLOWED_ROOT_MD)
    check(not extra, "no research documents at the root", str(extra) if extra else "")
    check(md == ALLOWED_ROOT_MD, "exactly README + ARCHITECTURE + CHECKPOINTS", str(sorted(md)))


def test_readme_answers_four_questions_first():
    print("\n=== 2. the first screen answers what / install / load / actions ===")
    text = (ROOT / "README.md").read_text()
    head = text[:text.index("---\n\n## Supported models")] if "## Supported models" in text else text
    at = [head.find(h) for h in FIRST_SCREEN]
    check(all(i > 0 for i in at), "install, load and get-actions all appear up front", str(at))
    check(at == sorted(at), "and in that order")
    check("Runtime.from_pretrained" in head, "the load example is the public Runtime API")
    check("episode.predict" in head, "the actions example loops")
    for banned in ("Optimizer(", "tier_ceiling", "port=", "plan.serve"):
        check(banned not in head, f"the first screen does not mention {banned!r}")


def test_readme_has_no_research_chronology():
    print("\n=== 3. no research chronology or profiling tables ===")
    text = (ROOT / "README.md").read_text()
    for banned in ("What's New", "Layer 6", "Layer 5", "P001", "P007", "forwards/cycle",
                   "marginal slope", "NOT EVALUATED", "ABBA"):
        check(banned not in text, f"README does not mention {banned!r}")
    # a date-stamped changelog line like "- **[2026/08]**" is the notebook tell
    check(not re.search(r"\*\*\[20\d\d/\d\d\]\*\*", text), "no date-stamped changelog entries")


def test_every_readme_link_resolves():
    print("\n=== 4. every link in the README points at something ===")
    text = (ROOT / "README.md").read_text()
    targets = [t for t in re.findall(r"\]\(([A-Za-z0-9_./-]+)\)", text) if not t.startswith("http")]
    missing = [t for t in targets if not (ROOT / t.split("#")[0]).exists()]
    check(not missing, f"all {len(targets)} local links resolve", str(missing) if missing else "")


def main() -> int:
    test_root_is_product_shaped()
    test_readme_answers_four_questions_first()
    test_readme_has_no_research_chronology()
    test_every_readme_link_resolves()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the root and the README present a maintained framework.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
