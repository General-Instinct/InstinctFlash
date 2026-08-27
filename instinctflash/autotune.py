"""Measured per-site autotune. Picks between implementations that already exist, by timing them.

A SITE is a place where two or more implementations of the same computation coexist, each
independently correct, and the right one depends on the device. The conv backend/layout choice
(P007) is one: NDHWC reaches cuDNN at 4.35-7.24x per signature on H100, and nothing about that
sentence is knowable on silicon it was not measured on. The vla2 MoE launch schedule is another:
64 batch-1 GEMM launches or 2 strided-batched launches over the same bytes.

WHAT THIS IS NOT. It is not search-over-schedules, and it must never become one. Candidates are
enumerated by a human; each carries a declared equivalence tier against the site's BASELINE and
names the evidence behind that claim. The runner's only degree of freedom is WHICH existing
candidate, decided by median wall time on this device -- the same stance as
`backends/conv/registry.select()` and `backends/registry.KernelRegistry.select()`, generalised
and given a persistent cache so a fleet does not re-time what it already knows.

TIER DISCIPLINE, same as passes. A swap between BITEXACT-equivalent candidates preserves the
plan tier. Anything else surfaces in the plan ("autotuned: <site> chose X over Y (1.3x),
equivalence NUMERIC") and respects the tier ceiling exactly like `Optimizer.compile` does: a
candidate above the ceiling is not benched, and the drop is recorded rather than silent.

THE CACHE is keyed by (device name + sm, model_id, site, shape signature) at
`~/.cache/instinctflash/autotune.json`. A hit skips the bench. A cached winner that the current
tier ceiling would not allow is ignored and re-measured, never trusted across a ceiling change.

OVERRIDES. `IFL_AUTOTUNE=0` disables every site (baseline serves, nothing benched, nothing
cached). `IFL_AUTOTUNE_<SITE>=<candidate>` forces one site. A forced choice is a preference,
not a measurement, so it is never written to the cache; forcing a candidate the tier ceiling
forbids raises, because silently exceeding the ceiling is the lie the ceiling exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from instinctflash.passes.contract import DeviceProfile, Tier


@dataclass(frozen=True)
class Candidate:
    """One implementation of one site.

    `tier` is the equivalence claim AGAINST THE SITE'S BASELINE -- what selecting this candidate
    does to the numbers, not how good the implementation is. `evidence` names the measurement or
    certificate backing the claim; a tier nobody can point at is a tier nobody should trust
    (the conv candidate cites its 555-episode paired non-inferiority certificate, the MoE
    schedule cites the M2 unit tests' maxabs_batched_vs_batch1_loop = 0.0).
    """

    name: str
    tier: Tier
    evidence: str
    params: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class Site:
    """A named choice point. Declarative on purpose: importable with no torch and no GPU."""

    name: str
    candidates: tuple[Candidate, ...]
    #: the incumbent -- selecting it changes nothing, so its tier is BITEXACT by definition,
    #: and it is what every short-circuit (disabled, no device, verify failure) falls back to.
    baseline: str
    #: part of the cache key: a decision measured at one shape must not answer another.
    shape_signature: str = ""

    def __post_init__(self):
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"site {self.name!r}: duplicate candidate names {names}")
        if self.baseline not in names:
            raise ValueError(f"site {self.name!r}: baseline {self.baseline!r} is not a "
                             f"candidate (have {names})")
        base = self.candidate(self.baseline)
        if base.tier is not Tier.BITEXACT:
            raise ValueError(
                f"site {self.name!r}: baseline {self.baseline!r} declares tier {base.tier.name}. "
                f"The baseline is the incumbent; selecting it changes nothing, so its tier "
                f"against itself is BITEXACT by definition. Declaring otherwise means the "
                f"baseline is not actually the incumbent.")

    def candidate(self, name: str) -> Candidate:
        for c in self.candidates:
            if c.name == name:
                return c
        raise KeyError(f"site {self.name!r} has no candidate {name!r}; "
                       f"have {[c.name for c in self.candidates]}")


@dataclass
class Decision:
    """What the runner chose, and on what basis. `reason` is the line explain() shows."""

    site: str
    chosen: str
    #: the best rejected alternative ("" when nothing was benched against the winner)
    over: str
    speedup: float
    #: tier of the SWAP: the chosen candidate's declared tier (BITEXACT when the baseline won)
    tier: Tier
    #: measured | cache | forced | disabled | ceiling | unopposed | no-device | verify-failed
    source: str
    timings_ms: dict = field(default_factory=dict)
    reason: str = ""


# --- site registry --------------------------------------------------------------------------

SITES: dict[str, Site] = {}


def register_site(site: Site) -> Site:
    """Idempotent for a re-declaration of the same choice; a site whose CANDIDATES or baseline
    differ under the same name raises, because two modules disagreeing about what a site is must
    not be resolved by import order. `shape_signature` is deliberately not part of that identity:
    it varies per instantiation (a 4-layer smoke engine and the 36-layer real one are the same
    site at different shapes) and it does its work in the cache key, not here."""
    prev = SITES.get(site.name)
    if prev is not None and (prev.candidates != site.candidates or prev.baseline != site.baseline):
        raise ValueError(f"site {site.name!r} is already registered with different candidates "
                         f"or baseline; rename one of them")
    SITES[site.name] = site
    return site


# --- cache ----------------------------------------------------------------------------------

def cache_path() -> Path:
    p = os.environ.get("IFL_AUTOTUNE_CACHE")
    if p:
        return Path(p)
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(root, "instinctflash", "autotune.json")


def _cache_key(device: DeviceProfile, model_id: str, site: Site) -> str:
    cap = f"sm{device.capability[0]}{device.capability[1]}"
    return "|".join((device.name, cap, model_id, site.name, site.shape_signature))


def _cache_load() -> dict:
    try:
        return json.loads(cache_path().read_text())
    except Exception:                                            # noqa: BLE001 - absent or corrupt
        return {}                                                # both mean "measure again"


def _cache_store(key: str, entry: dict) -> None:
    """Read-modify-write with an atomic rename, so a crashed process cannot half-write the file
    every later load would then fail to parse."""
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _cache_load()
    doc[key] = entry
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --- overrides ------------------------------------------------------------------------------

def _env_name(site_name: str) -> str:
    return "IFL_AUTOTUNE_" + re.sub(r"[^A-Za-z0-9]", "_", site_name).upper()


def _disabled() -> bool:
    return os.environ.get("IFL_AUTOTUNE", "1").strip().lower() in ("0", "off", "false", "no")


# --- the runner -----------------------------------------------------------------------------

def _reason(site: Site, chosen: str, over: str, speedup: float, tier: Tier, source: str,
            extra: str = "") -> str:
    if over:
        line = (f"autotuned: {site.name} chose {chosen} over {over} ({speedup:.2f}x), "
                f"equivalence {tier.name} [{source}]")
    else:
        line = f"autotuned: {site.name} kept {chosen}, equivalence {tier.name} [{source}]"
    return line + (f" -- {extra}" if extra else "")


def autotune(site: Site, bench: Callable[[Candidate], float], *, model_id: str = "?",
             device: DeviceProfile | None = None, tier_ceiling: Tier = Tier.BITEXACT,
             verify: Callable[[str], float] | None = None,
             n: int = 5, warmup: int = 2, use_cache: bool = True) -> Decision:
    """Pick a candidate for `site` on this device. Returns a Decision, never raises on a slow
    or crashing candidate -- a candidate that cannot run loses, loudly, and the baseline always
    exists to fall back to.

    `bench(candidate) -> ms` measures one invocation-batch of the candidate however the site
    does that honestly (CUDA events and a sync for GPU work). The runner adds the protocol:
    `warmup` unrecorded calls, then `n` recorded ones, MEDIAN taken -- median rather than mean
    because a single clock-ramp outlier must not pick a winner.

    `verify(candidate_name) -> max|delta|` is the audit_tier stance applied here: a candidate
    CLAIMING BITEXACT equivalence is checked against a measured delta before the swap is
    trusted, and a nonzero delta keeps the baseline rather than serving a claim that just
    failed. Verification runs when measuring; a cache hit reuses a decision that was verified
    when it was made.

    Raising `tier_ceiling` is a claim budget, not a speed knob -- identical wording to
    `Runtime.from_pretrained`, because it is the same contract.
    """
    baseline = site.candidate(site.baseline)

    if _disabled():
        return Decision(site.name, baseline.name, "", 1.0, Tier.BITEXACT, "disabled",
                        reason=_reason(site, baseline.name, "", 1.0, Tier.BITEXACT, "disabled",
                                       "IFL_AUTOTUNE=0"))

    forced = os.environ.get(_env_name(site.name))
    if forced:
        cand = site.candidate(forced.strip())                    # unknown name raises, with names
        if cand.tier > tier_ceiling:
            raise RuntimeError(
                f"{_env_name(site.name)}={forced!r} forces a {cand.tier.name} candidate under a "
                f"{tier_ceiling.name} tier ceiling. Those are contradictory instructions, and "
                f"silently exceeding the ceiling is the lie it exists to prevent. Raise the "
                f"ceiling explicitly, or force a candidate within it.")
        return Decision(site.name, cand.name, "", 1.0, cand.tier, "forced",
                        reason=_reason(site, cand.name, "", 1.0, cand.tier, "forced",
                                       f"{_env_name(site.name)}={forced}"))

    dropped = [c for c in site.candidates if c.tier > tier_ceiling]
    eligible = [c for c in site.candidates if c.tier <= tier_ceiling]
    ceiling_note = (f"{[c.name for c in dropped]} excluded by tier ceiling {tier_ceiling.name}"
                    if dropped else "")

    if len(eligible) == 1:
        only = eligible[0]
        return Decision(site.name, only.name, "", 1.0, only.tier, "unopposed",
                        reason=_reason(site, only.name, "", 1.0, only.tier, "unopposed",
                                       ceiling_note or "no other candidate"))

    if device is None:
        try:
            device = DeviceProfile.probe()
        except Exception:                                        # noqa: BLE001 - no torch, no CUDA
            device = None
    if device is None:
        return Decision(site.name, baseline.name, "", 1.0, Tier.BITEXACT, "no-device",
                        reason=_reason(site, baseline.name, "", 1.0, Tier.BITEXACT, "no-device",
                                       "no device probed; a measured choice needs a "
                                       "measured target, so the incumbent stands"))

    key = _cache_key(device, model_id, site)
    if use_cache:
        hit = _cache_load().get(key)
        if hit and hit.get("chosen") in {c.name for c in eligible}:
            chosen = site.candidate(hit["chosen"])
            timings = dict(hit.get("timings_ms") or {})
            over, speedup = _runner_up(chosen.name, timings)
            return Decision(site.name, chosen.name, over, speedup, chosen.tier, "cache",
                            timings_ms=timings,
                            reason=_reason(site, chosen.name, over, speedup, chosen.tier,
                                           "cache", ceiling_note))
        # A stale/foreign cached winner (renamed candidate, or one the current ceiling forbids)
        # is ignored and re-measured, never trusted across a ceiling change.

    # -- verify claimed bit-identity before trusting it (audit_tier: "this is not optional") --
    if verify is not None:
        kept, refused = [], []
        for c in eligible:
            if c.name == baseline.name or c.tier is not Tier.BITEXACT:
                kept.append(c)
                continue
            try:
                delta = float(verify(c.name))
            except Exception as e:                               # noqa: BLE001 - cannot verify -> refuse
                refused.append((c.name, f"verify raised {type(e).__name__}: {e}"))
                continue
            if delta != 0.0:
                refused.append((c.name, f"claimed BITEXACT-equivalent, measured "
                                        f"max|delta| = {delta:.3e}; DEMOTED, swap refused"))
            else:
                kept.append(c)
        eligible = kept
        if refused and len(eligible) == 1:
            detail = "; ".join(f"{n_}: {why}" for n_, why in refused)
            return Decision(site.name, baseline.name, "", 1.0, Tier.BITEXACT, "verify-failed",
                            reason=_reason(site, baseline.name, "", 1.0, Tier.BITEXACT,
                                           "verify-failed", detail))

    # -- the micro-bench: warmup, n reps, median ----------------------------------------------
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    for c in eligible:
        try:
            for _ in range(warmup):
                bench(c)
            timings[c.name] = statistics.median(bench(c) for _ in range(n))
        except Exception as e:                                   # noqa: BLE001 - a candidate that
            failures[c.name] = f"{type(e).__name__}: {e}"        # cannot run loses, loudly
            timings[c.name] = float("inf")
    if all(t == float("inf") for t in timings.values()):
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        return Decision(site.name, baseline.name, "", 1.0, Tier.BITEXACT, "verify-failed",
                        timings_ms=timings,
                        reason=_reason(site, baseline.name, "", 1.0, Tier.BITEXACT,
                                       "verify-failed", f"every candidate failed to bench: "
                                                        f"{detail}"))

    # deterministic: min() over (median, declaration order) -- a tie goes to the earlier
    # declaration, and sites declare the baseline first, so a tie changes nothing.
    order = {c.name: i for i, c in enumerate(site.candidates)}
    winner_name = min(timings, key=lambda k: (timings[k], order[k]))
    chosen = site.candidate(winner_name)
    over, speedup = _runner_up(winner_name, timings)
    extra = ceiling_note
    if failures:
        fail_note = "; ".join(f"{k} FAILED: {v}" for k, v in failures.items())
        extra = f"{extra}; {fail_note}" if extra else fail_note

    if use_cache:
        _cache_store(key, {
            "chosen": chosen.name,
            "timings_ms": {k: v for k, v in timings.items() if v != float("inf")},
            "tier": chosen.tier.name,
            "model_id": model_id,
            "device": f"{device.name} sm{device.capability[0]}{device.capability[1]}",
            "protocol": f"median of {n} after {warmup} warmup",
        })
    return Decision(site.name, chosen.name, over, speedup, chosen.tier, "measured",
                    timings_ms=dict(timings),
                    reason=_reason(site, chosen.name, over, speedup, chosen.tier, "measured",
                                   extra))


def _runner_up(winner: str, timings: Mapping[str, float]) -> tuple[str, float]:
    """The best rejected alternative and the speedup over it. ('', 1.0) when unopposed."""
    rest = {k: v for k, v in timings.items() if k != winner and v != float("inf")}
    if not rest or not timings.get(winner):
        return "", 1.0
    over = min(rest, key=rest.get)                               # nearest competitor, honestly:
    w = timings[winner]                                          # the margin quoted is the
    return over, (rest[over] / w) if w else float("nan")         # smallest one, not the flashiest


# --- plan surfacing -------------------------------------------------------------------------

def record_decision(plan, decision: Decision) -> None:
    """Append the decision to a Plan so explain() shows it and Plan.tier() prices it.

    `applies` is True only for an actual swap away from the baseline, and `tier` is the swap's
    equivalence tier -- so a BITEXACT-equivalent swap preserves the plan tier and a NUMERIC one
    makes the plan NUMERIC, by the existing max() rule with no new machinery. A kept baseline
    is recorded with applies=False rather than omitted: a reader must be able to see that the
    site was evaluated and what it would have taken to swap.
    """
    results = getattr(plan, "results", None)
    if not isinstance(results, list):
        return
    from instinctflash.planners.planner import PassResult

    site = SITES.get(decision.site)
    chosen = site.candidate(decision.chosen) if site is not None else None
    name = f"autotune:{decision.site}"
    results[:] = [r for r in results if r.name != name]          # re-tunes replace, not stack
    results.append(PassResult(
        name=name,
        applies=(site is not None and decision.chosen != site.baseline),
        tier=decision.tier,
        reason=decision.reason,
        params={
            "site": decision.site,
            "chosen": decision.chosen,
            "source": decision.source,
            "timings_ms": dict(decision.timings_ms),
            **({"evidence": chosen.evidence} if chosen is not None else {}),
        },
        expected_win=(f"{decision.speedup:.2f}x at this site (measured)"
                      if decision.source in ("measured", "cache") and decision.over
                      else "unknown"),
    ))


__all__ = ["Candidate", "Site", "Decision", "SITES", "register_site", "autotune",
           "record_decision", "cache_path"]
