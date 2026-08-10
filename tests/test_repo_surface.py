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
    # the first screen ends at whichever secondary section comes first
    ends = [text.index(h) for h in ("## What's new", "## Supported models") if h in text]
    head = text[:min(ends)] if ends else text
    at = [head.find(h) for h in FIRST_SCREEN]
    check(all(i > 0 for i in at), "install, load and get-actions all appear up front", str(at))
    check(at == sorted(at), "and in that order")
    check("Runtime.from_pretrained" in head, "the load example is the public Runtime API")
    check("episode.predict" in head, "the actions example loops")
    for banned in ("Optimizer(", "tier_ceiling", "port=", "plan.serve"):
        check(banned not in head, f"the first screen does not mention {banned!r}")


def test_readme_has_no_research_chronology():
    print("\n=== 3. no research chronology or profiling tables ===")
    # A "What's new" section IS allowed, and is wanted: every widely used framework has one. What is
    # not allowed is what the old one had -- optimization milestones, pass numbers, dated research
    # entries. The distinction is user-facing capability versus internal chronology, so the check is
    # on the vocabulary of chronology rather than on the heading.
    text = (ROOT / "README.md").read_text()
    for banned in ("Layer 6", "Layer 5", "P001", "P007", "marginal slope", "NOT EVALUATED", "ABBA",
                   "RETRACTED", "PROPOSED API", "operating point", "Quality (25"):
        check(banned not in text, f"README does not mention {banned!r}")
    check(not re.search(r"\*\*\[20\d\d/\d\d\]\*\*", text), "no date-stamped changelog entries")
    news = re.search(r"^## What's new", text, re.M)
    if news:
        body = text[news.end():text.index("\n## ", news.end())]
        check("×" not in body and "ms" not in body.split(), "the news section is not a results table")


def test_canonical_docs_describe_only_main():
    print("\n=== 5. the canonical docs describe main, not its history ===")
    # Every one of these is something the docs actually said while describing a system that had
    # moved on: a proposed API, an inline retraction, a pass list that graph capture was still part
    # of, and two contradictory forwards-per-cycle figures in the same file.
    for name in ("README.md", "ARCHITECTURE.md", "CHECKPOINTS.md"):
        text = (ROOT / name).read_text()
        for banned in ("PROPOSED API", "RETRACTED", "not implemented yet", "PROFILE.md",
                       "3.38", "Fast runs", "566 matched pairs"):
            check(banned not in text, f"{name} does not contain {banned!r}")
        dead = [t for t in re.findall(r"\]\(([A-Za-z0-9_./-]+\.md)", text)
                if not (ROOT / t).exists()]
        check(not dead, f"{name} links no deleted document", str(dead) if dead else "")


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
    test_canonical_docs_describe_only_main()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the root and the README present a maintained framework.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
