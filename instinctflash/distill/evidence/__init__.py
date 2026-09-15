"""The distill evidence base: the campaigns' measured frontiers, registered as DATA.

Each file is a frontier in the sweep-report row shape (`benchmarks.vla.schedule_sweep
.build_sweep_report`: `baseline`, `rows[]` with nfe / guidance / point_success / delta / interval
/ deciding_lower_bound / n_pairs / forwards_batch1 / forwards_batch2 / cycle_p50_ms, plus
`noise_floor`), so `instinctflash.distill.screen_verdict` and the control gate consume campaign
evidence and product sweeps identically. Every row cites the archive it was read from; the files
are produced by `iwm_distill/fewstep/build_evidence_files.py` from the campaign analyses, never
by hand, and never re-analyse episodes.

WHY DATA AND NOT PROSE. The pi05 adapter's docstring already carried E5 and E9; the wan_va
adapter carried "the untrained cliff is 2->1". Those sentences were true when written and one of
them is now known to be a guidance artifact. A frontier as data can be re-read by the verdict
code, compared to a new sweep, and cited row by row; a sentence cannot. `known.py` stays what it
is — execution declarations the runtime reads — and carries none of this.

    wan_va_robotwin_guidance_frontier.json   LingBot-VA on RoboTwin: the untrained (schedule x
                                             guidance) frontier over 2V/4A, 3V/4A, 2V/2A, 1V/4A,
                                             1V/2A, 1V/1A at w in {1, 3, 5, 7, 9} (where run),
                                             paired vs the rerun 2V/4A@w5 baseline arm; the H1
                                             trained arms for the record (never frontier rows).
    pi05_libero_schedule_frontier.json       pi05 v044 on LIBERO (4 suites, 160 pairs/point):
                                             the untrained schedule frontier nfe 10 -> 1, no
                                             guidance axis (the family has none), run through
                                             benchmarks.vla.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

FILES = {
    "wan_va": "wan_va_robotwin_guidance_frontier.json",
    "pi05": "pi05_libero_schedule_frontier.json",
}


def evidence_path(family: str) -> Path:
    if family not in FILES:
        raise KeyError(f"no registered evidence for family {family!r}; registered: {sorted(FILES)}")
    return _HERE / FILES[family]


def load_evidence(family: str) -> dict[str, Any]:
    """The family's registered frontier (a sweep-report-shaped document)."""
    doc = json.loads(evidence_path(family).read_text())
    if doc.get("kind") != "fewstep_evidence_frontier" or doc.get("family") != family:
        raise ValueError(f"{evidence_path(family)}: not a fewstep_evidence_frontier for {family!r}")
    return doc


def registered_families() -> list[str]:
    return sorted(FILES)
