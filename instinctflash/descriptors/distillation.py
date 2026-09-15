"""The `provenance.distillation` block: what a distilled checkpoint must be able to prove.

A few-step distilled checkpoint is the one artifact in this repository whose headline claim is
easiest to state dishonestly: "the student matches the teacher at 1/8th the forwards" conflates
two effects -- what step reduction alone cost, and what training bought back -- and the
literature routinely reports the pair as one number (methods memo §2.1: Flash-WAM's sim tables,
the PDD paper, LingBot-VA 2.0, OFP all omit the untrained matched-NFE control). This block is
where a checkpoint carries the separation.

THE MATCHED-NFE-CONTROL LAW (campaign law, `docs/rfc/fewstep-distillation.md`):

    No trained few-step result is reportable unless it is paired, on pinned scenes, against the
    UNTRAINED teacher run at the identical execution configuration, and the report states B-A
    and C-B separately (A = full teacher, B = untrained matched-NFE control, C = student).

The block therefore REQUIRES a `matched_nfe_control` object. A checkpoint mid-campaign that has
trained but not yet run its control stamps the control fields as `"FILL_ME"` -- the scaffold's
sentinel, enforced the scaffold's way: `instinctflash validate` flags every sentinel as a
PROBLEM and exits non-zero until the control evidence exists. A distillation block with no
control at all is refused outright; that absence is the exact failure the law exists to catch.

INTEGRITY, LIKE THE CERTIFICATE BLOCK. The block carries sha256 hashes of the things it claims
(teacher weights, dataset, outcome files) and a self-hash over its own content, so a hand-edited
delta is detected by every later plain `validate` -- same discipline, same reason as
`provenance.certificate` (tests/test_validate_certificate.py).

NAMESPACE. This is provenance: the runtime NEVER reads it (descriptors/checkpoint.py enforces
the two-namespace split, and "distillation" is on the execution blacklist). What the runtime
needs from a distilled checkpoint -- nfe, guidance mode, output projection -- is declared in
`execution` exactly as for any other checkpoint; HOW those facts came to be true lives here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

FILL_ME = "FILL_ME"

#: Top-level keys a distillation block must carry. `content_sha256` is added by the builder.
REQUIRED_KEYS = (
    "recipe_id",            # which recipe produced the student, e.g. "fewstep_video_cd_v1"
    "family",               # the per-family adapter that instantiated it, e.g. "wan_va"
    "teacher",              # {model_id, weights_sha256} -- hash of what was distilled FROM
    "dataset",              # {id, sha256} -- hash of what it was distilled ON
    "schedule",             # {nfe: {phase: int}, grids: {phase: [t...]}, guidance: {stream:
                            #   {mode, scale}}} -- the OPERATING POINT the student was trained
                            #   at (RFC §11: schedule grid, per-stream guidance, CFG batching);
                            #   the servable copies live in execution.nfe / execution.guidance
    "matched_nfe_control",  # the law. see CONTROL_KEYS
    "stamped_at",
)

#: What the control object must carry once the control evidence exists. Until then each value
#: is the FILL_ME sentinel and validate keeps failing (never silently passing) the package.
CONTROL_KEYS = (
    "teacher_outcomes_sha256",   # arm A: the full teacher, paired episodes
    "control_outcomes_sha256",   # arm B: UNTRAINED teacher at the student's exact schedule
    "student_outcomes_sha256",   # arm C: the trained student
    "b_minus_a",                 # cost of step reduction  {delta, interval, n_pairs}
    "c_minus_b",                 # what training bought    {delta, interval, n_pairs}
    # THE STRONG FORM OF THE LAW (RFC §11): arm B is the BEST untrained configuration over the
    # guidance grid swept at the student's schedule, not the shipped guidance. The H1 screen is
    # the precedent: vs 1V/4A@w5 the student showed +14.3 pp; vs the best untrained knob
    # (1V/4A@w3) it showed +0.0. So the block states which guidance the control ran at and which
    # grid it was chosen from; a control whose grid was not swept is refused by verify_point.
    "control_guidance",          # {stream: {mode, scale}} the control arm served
    "guidance_grid_swept",       # [{guidance, success, point}, ...] the configurations screened
)


def content_hash(block: Mapping[str, Any]) -> str:
    """Self-hash over everything except the hash itself -- the certificate block's rule."""
    payload = {k: v for k, v in block.items() if k != "content_sha256"}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arm_summary(certificate: Mapping[str, Any] | Any) -> dict[str, Any]:
    """A paired-comparison summary from a `verify.certify` Certificate (or its dict form)."""
    if hasattr(certificate, "to_json"):
        certificate = json.loads(certificate.to_json())
    return {
        "delta": certificate["delta"],
        "ci95": list(certificate["ci95"]),
        "n_pairs": certificate["n_pairs"],
        "discordant": list(certificate["discordant"]),
        "verdict": certificate["verdict"],
        "margin_declared": certificate["margin_declared"],
    }


def build_block(
    *,
    recipe_id: str,
    family: str,
    teacher_model_id: str,
    teacher_weights_sha256: str,
    dataset_id: str,
    dataset_sha256: str,
    schedule: Mapping[str, Any],
    control: Mapping[str, Any] | None = None,
    stamped_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a distillation block; missing control evidence becomes FILL_ME, never absence.

    `control`, when given, must carry every key in CONTROL_KEYS (built by
    `instinctflash.distill.pipeline.verify_point`, which is the only path that can produce a
    complete one because it refuses to run without the matched control). When None, the block
    is stamped with sentinels so the package loudly fails validation until the control runs.
    """
    from datetime import datetime, timezone

    if control is not None:
        missing = sorted(set(CONTROL_KEYS) - set(control))
        if missing:
            raise ValueError(
                f"matched_nfe_control is missing {missing}. A partial control is not a control: "
                "B-A and C-B must both be present, or stamp with control=None and let the "
                "FILL_ME sentinels keep the package failing validation until the evidence exists."
            )
        control_block = dict(control)
    else:
        control_block = {key: FILL_ME for key in CONTROL_KEYS}
    block: dict[str, Any] = {
        "recipe_id": recipe_id,
        "family": family,
        "teacher": {"model_id": teacher_model_id, "weights_sha256": teacher_weights_sha256},
        "dataset": {"id": dataset_id, "sha256": dataset_sha256},
        "schedule": json.loads(json.dumps(dict(schedule))),
        "matched_nfe_control": control_block,
        "stamped_at": stamped_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        overlap = sorted(set(extra) & set(block))
        if overlap:
            raise ValueError(f"extra keys {overlap} would shadow required block fields")
        block.update(extra)
    block["content_sha256"] = content_hash(block)
    return block


def stamp_distillation(pkg_dir: str | Path, block: Mapping[str, Any]) -> Path:
    """Write the block into the package's provenance, atomically, certificate-style."""
    from instinctflash.cli_config import _atomic_write
    from instinctflash.descriptors.checkpoint import _declaration_file

    decl = _declaration_file(Path(pkg_dir))
    if decl is None:
        raise FileNotFoundError(
            f"cannot stamp a distillation block: {pkg_dir} has no declaration file "
            "(instinctflash.json) to carry a provenance block"
        )
    doc = json.loads(decl.read_text())
    doc.setdefault("provenance", {})["distillation"] = dict(block)
    _atomic_write(decl, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return decl


def _fill_me_paths(prefix: str, value: Any) -> list[str]:
    if isinstance(value, str) and value == FILL_ME:
        return [prefix]
    if isinstance(value, Mapping):
        return [p for k, v in value.items() for p in _fill_me_paths(f"{prefix}.{k}", v)]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in _fill_me_paths(f"{prefix}[{i}]", v)]
    return []


def verify_distillation(pkg_dir: str | Path) -> tuple[str, dict | None, list[str]]:
    """-> (status, block, problems): status is 'absent', 'intact' or 'tampered'.

    Called by plain `instinctflash validate` on every run. `problems` is non-empty when the
    block is structurally dishonest even though un-tampered: required fields missing, the
    matched control absent or partial, or FILL_ME sentinels still standing in. Each problem
    line says why it blocks, in the block's own vocabulary.
    """
    from instinctflash.descriptors.checkpoint import _declaration_file

    decl = _declaration_file(Path(pkg_dir))
    if decl is None:
        return "absent", None, []
    try:
        doc = json.loads(decl.read_text())
    except Exception:  # noqa: BLE001 - the structure check reports unparseable declarations
        return "absent", None, []
    block = (doc.get("provenance") or {}).get("distillation")
    if not isinstance(block, dict):
        return "absent", None, []
    if block.get("content_sha256") != content_hash(block):
        return "tampered", block, [
            "provenance.distillation fails its integrity hash — the block has been edited "
            "after stamping"
        ]

    problems: list[str] = []
    missing = sorted(set(REQUIRED_KEYS) - set(block))
    if missing:
        problems.append(
            f"provenance.distillation is missing {missing} — a distilled checkpoint must state "
            "what it was distilled from, on, at which schedule, and against which control"
        )
    control = block.get("matched_nfe_control")
    if "matched_nfe_control" not in (missing or ()) and not isinstance(control, Mapping):
        problems.append(
            "provenance.distillation.matched_nfe_control is not an object — the matched-NFE-"
            "control law admits no trained few-step claim without its untrained control"
        )
        control = None
    if isinstance(control, Mapping):
        control_missing = sorted(set(CONTROL_KEYS) - set(control))
        if control_missing:
            problems.append(
                f"matched_nfe_control is missing {control_missing} — B−A (cost of step "
                "reduction) and C−B (what training bought) must be stated separately"
            )
    for path in _fill_me_paths("provenance.distillation", block):
        problems.append(
            f'{path} is "FILL_ME" — the control evidence for this trained point does not '
            "exist yet; run the matched-NFE control (screening rows from a schedule sweep, or "
            "a fresh paired run at the identical schedule) and re-stamp"
        )
    return "intact", block, problems
