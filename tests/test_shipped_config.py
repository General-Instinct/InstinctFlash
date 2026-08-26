#!/usr/bin/env python3
"""One shipped configuration, asserted everywhere it is written down.

THE DRIFT THIS EXISTS TO CATCH, because it already happened. `verify/released.py` listed seven passes
as released and `is_verified() == True`, while the certification launch line enabled three. P005 was
registered at 1.380x BITEXACT while `serve_variant.py`'s own `--graph-blocks` help called it NOT
SHIPPABLE and a later measurement put capture 1.43x SLOWER. And P007 -- the only pass that moves the
current cycle, certified on 555 paired episodes -- was in no launch script at all.

Nothing was lying; the facts were true when written and drifted apart afterwards. So the fix is not a
one-time correction, it is a derivation: `shipped_configuration()` is the source, every other place
derives, and this test fails when they disagree.

No GPU, no torch, no model. Pure text and registry.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instinctflash.verify.released import (  # noqa: E402
    AVAILABLE, DISPOSITIONS, NOT_RECOMMENDED, RELEASED, SERVED, disposition_of,
    served_tier, shipped_configuration, shipped_pids,
)

LAUNCHERS = ("eval/lingbot_va_robotwin/run_pdd_cert.sh", "eval/lingbot_va_robotwin/run_sweep.sh")
SERVE = "eval/lingbot_va_robotwin/serve_variant.py"

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def test_every_released_pass_has_a_disposition():
    print("\n=== 1. the ledger and the disposition table cover the same passes ===")
    led = {r.pid for r in RELEASED}
    dis = {d.pid for d in DISPOSITIONS}
    check(led == dis, "every released pass has exactly one disposition",
          f"ledger-only {sorted(led - dis)}, disposition-only {sorted(dis - led)}")
    check(len({d.pid for d in DISPOSITIONS}) == len(DISPOSITIONS), "no duplicate dispositions")
    for d in DISPOSITIONS:
        check(d.status in (SERVED, AVAILABLE, NOT_RECOMMENDED), f"{d.pid} status is a known value",
              d.status)
        check(bool(d.note.strip()), f"{d.pid} states WHY it holds that status today")


def test_launch_scripts_match_the_registry():
    print("\n=== 2. THE ONE THAT CAUGHT THE REAL BUG: launchers == shipped_configuration() ===")
    want = shipped_configuration()
    print(f"  registry says: {' '.join(want)}")
    for rel in LAUNCHERS:
        text = (ROOT / rel).read_text()
        for flag in want:
            check(flag in text, f"{Path(rel).name} enables {flag}")
        # and nothing that is NOT recommended may appear on a launch line
        for d in DISPOSITIONS:
            if d.status is not NOT_RECOMMENDED:
                continue
            for flag in d.serving_flags:
                # the flag may appear in a comment; only a launch line is a problem
                on_cmd = any(flag in ln and not ln.strip().startswith("#")
                             for ln in text.splitlines())
                check(not on_cmd, f"{Path(rel).name} does NOT enable {flag} ({d.pid}, "
                                  f"NOT RECOMMENDED)")


def test_serve_variant_defines_every_shipped_flag():
    print("\n=== 3. every shipped flag actually exists on serve_variant.py ===")
    text = (ROOT / SERVE).read_text()
    for flag in shipped_configuration():
        check(f'"{flag}"' in text, f"serve_variant.py defines {flag}")


def test_not_recommended_passes_say_so_where_a_reader_will_look():
    print("\n=== 4. a NOT RECOMMENDED pass warns at the point of use ===")
    text = (ROOT / SERVE).read_text()
    for d in DISPOSITIONS:
        if d.status is not NOT_RECOMMENDED:
            continue
        for flag in d.serving_flags:
            i = text.find(f'"{flag}"')
            if not check(i >= 0, f"serve_variant.py defines {flag}"):
                continue
            helptext = text[i:i + 1400].upper()
            check(any(w in helptext for w in ("NOT SHIPPABLE", "NOT RECOMMENDED", "DO NOT ENABLE",
                                              "MEASUREMENT ONLY")),
                  f"{flag} help warns the reader ({d.pid})")


def test_readme_states_the_same_configuration():
    print("\n=== 5. the README architecture chapter describes the shipped configuration, THIS one ===")
    # This used to assert against the README. The README is now the product front page -- what
    # InstinctFlash is, how to install it, how to load a model, how to get actions -- and a table of
    # serving CLI flags is not that. The invariant is unchanged and still enforced: whatever document
    # states the shipped configuration must state THIS one.
    # 2026-08-26: the front-page README sheds serving-flag detail (user decision — the front page
    # is install/load/actions/results). The invariant is unchanged: whatever document states the
    # shipped configuration must state THIS one. That document is now the LingBot eval README.
    text = (ROOT / "eval" / "lingbot_va_robotwin" / "README.md").read_text()
    check("shipped_configuration()" in text or "SHIPPED CONFIGURATION" in text.upper(),
          "the LingBot eval README has a shipped-configuration section")
    for flag in shipped_configuration():
        check(flag in text, f"the shipped-config doc lists {flag}")
    for d in DISPOSITIONS:
        if d.status is NOT_RECOMMENDED:
            for pid in (d.pid,):
                check(pid in text, f"the shipped-config doc accounts for {pid} (NOT RECOMMENDED)")


def test_served_tier_is_stated_honestly():
    print("\n=== 6. the served chain's tier is the weakest link ===")
    tiers = {r.pid: r.tier.name for r in RELEASED if disposition_of(r.pid).status is SERVED}
    print(f"  served: {tiers}")
    expect = "NUMERIC" if "NUMERIC" in tiers.values() else "BITEXACT"
    check(served_tier().name == expect, f"served_tier() is {expect}, not the best member",
          served_tier().name)
    if served_tier().name != "BITEXACT":
        for r in RELEASED:
            if disposition_of(r.pid).status is SERVED and r.tier.name != "BITEXACT":
                check(bool(r.certificate.strip()),
                      f"{r.pid} is served and non-BITEXACT, so it must carry a certificate")




def test_every_pass_module_is_classified():
    """A pass module is runtime code, a negative result, or an archived implementation. No fourth kind.

    Without this, an unshipped pass is indistinguishable from a shipped one by reading the tree -- which
    is how `forward_scratch` and `static_partition_hoist` sat next to `ring_kv` looking equally live.
    """
    print("\n=== 7. every pass module declares runtime status ===")
    served_names = {r.name for r in RELEASED}
    # modules that back a registry entry, by the pass's own module name
    registry_backed = {
        "substrate", "conditioning_prefill", "hoist_invariant_casts", "graph_capture",
        "stable_pools", "ring_kv", "conv_layout_ndhwc",
    }
    d = ROOT / "instinctflash" / "passes" / "lingbot"
    for f in sorted(d.glob("*.py")):
        if f.stem == "__init__":
            continue
        head = f.read_text().split('"""')[1] if '"""' in f.read_text() else ""
        classified = "STATUS: NEGATIVE RESULT" in head or "STATUS: ARCHIVED IMPLEMENTATION" in head
        check(classified or f.stem in registry_backed,
              f"{f.stem} is registry-backed or declares STATUS", "" if classified else "unclassified")


def main() -> int:
    print(f"shipped: {', '.join(shipped_pids())}  tier {served_tier().name}")
    test_every_released_pass_has_a_disposition()
    test_launch_scripts_match_the_registry()
    test_serve_variant_defines_every_shipped_flag()
    test_not_recommended_passes_say_so_where_a_reader_will_look()
    test_readme_states_the_same_configuration()
    test_served_tier_is_stated_honestly()
    test_every_pass_module_is_classified()
    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: the registry, both launch scripts, serve_variant and the README agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
