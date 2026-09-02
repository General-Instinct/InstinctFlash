"""The guidance half of an operating point: per-stream (mode, scale), and the CFG batching it implies.

AN OPERATING POINT IS A TUPLE, NOT A STEP COUNT. The few-step campaign (RFC
docs/rfc/fewstep-distillation.md §11) measured the same LingBot-VA schedule, 1V/4A, at four
guidance scales on the same pinned scenes: w=5 (shipped) 0.752, w=3 0.882, w=1 (positive-only,
batch-1) 0.883, w=9 0.276. `nfe` alone therefore underspecifies quality by up to 60 points; what a
checkpoint declares, what a plan prices, what a sweep varies and what a control is matched on is

    (schedule grid, per-stream guidance scale, CFG batching)

and this module is the one place the guidance leg is parsed, resolved and canonicalised, so every
consumer -- the runtime that writes the server's scale, the planner that prices batch-1 vs batch-2
forwards, the scaffold that inherits it, the sweep that varies it, the control gate that matches
on it -- reads the same tuple.

THREE DECLARATION FORMS, ONE MEANING. `execution.guidance` maps a stream to

    "cfg" | "positive_only" | "none"              a MODE; the scale is the family's own default
                                                   and is recorded as INHERITED, never invented
    3.0  (or "3.0" from a CLI)                     a SCALE; the mode is cfg (a scale IS a CFG
                                                   statement); <= 1 means the negative branch
                                                   is never requested
    {"mode": "cfg", "scale": 3.0}                  both, explicitly

The string form is the pre-campaign schema and stays valid forever (every built-in declaration
uses it). A contradiction -- `positive_only` or `none` WITH a scale above 1 -- is refused: a
positive-only stream discards its negative branch, so a scale > 1 would silently turn a combine
on that the mode says is off.

CFG BATCHING IS DERIVED, NEVER DECLARED. A stream requests its negative branch iff mode is cfg and
the served scale is above 1 (the LingBot server's own test, `guidance_scale > 1`). A forward is
batch-2 iff it touches a stream that requests one. Scale 1.0 -- or mode positive_only -- means
batch-1 for that stream's forwards, which is exactly the untrained operating point the campaign
found free (h1_report.md §4b: 1V/4A@w1 = 0.885 vs 0.752 at w=5, nine batch-1 forwards).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

MODES = ("cfg", "positive_only", "none")

#: Modes under which the negative branch is never combined, so the served scale is 1 by definition.
_NO_NEGATIVE_BRANCH = ("positive_only", "none")

ACCEPTED_FORMS = (
    'a mode name ("cfg" | "positive_only" | "none"), a numeric scale (e.g. 3.0), or '
    '{"mode": <mode>, "scale": <number>}'
)


class GuidanceDeclarationError(ValueError):
    """A guidance declaration that cannot be served as written."""


@dataclass(frozen=True)
class DeclaredGuidance:
    """One stream's guidance as DECLARED: parsed, validated, not yet resolved against a family."""

    mode: str | None    # None: the declaration named only a scale
    scale: float | None  # None: the declaration named only a mode
    form: str            # "mode" | "scale" | "mode+scale"

    def canonical(self) -> dict[str, Any]:
        return {"mode": self.mode, "scale": self.scale}


def parse_declared(value: Any, *, where: str = "guidance") -> DeclaredGuidance:
    """Parse one stream's declared guidance value. Refuses garbage loudly.

    A CLI flag arrives as a STRING, so numeric strings ("3", "1.0") are scales -- before this
    existed `--guidance video=3` fell through every case and applied nothing (found 2026-08-31
    preparing the guidance x NFE sweep: the served scale stayed at the model default while the
    operator believed it was re-tuned).
    """
    if isinstance(value, bool):
        raise GuidanceDeclarationError(f"{where}: a boolean is not a guidance declaration; "
                                       f"accepted: {ACCEPTED_FORMS}")
    if isinstance(value, (int, float)):
        return DeclaredGuidance(mode=None, scale=_scale(value, where), form="scale")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in MODES:
            return DeclaredGuidance(mode=text, scale=None, form="mode")
        try:
            number = float(text)
        except ValueError:
            raise GuidanceDeclarationError(
                f"{where}: {value!r} is neither a guidance mode nor a numeric scale; accepted: "
                f"{ACCEPTED_FORMS}. Serving something other than what the declaration says is the "
                f"failure this schema exists to prevent, so an unreadable value is refused rather "
                f"than ignored.") from None
        return DeclaredGuidance(mode=None, scale=_scale(number, where), form="scale")
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"mode", "scale"})
        if unknown:
            raise GuidanceDeclarationError(f"{where}: unknown keys {unknown}; accepted: {ACCEPTED_FORMS}")
        mode = value.get("mode")
        scale = value.get("scale")
        if mode is None and scale is None:
            raise GuidanceDeclarationError(f"{where}: an empty object declares nothing; accepted: "
                                           f"{ACCEPTED_FORMS}")
        if mode is not None:
            if not isinstance(mode, str) or mode.strip().lower() not in MODES:
                raise GuidanceDeclarationError(
                    f"{where}: mode {mode!r} is not one of {MODES}")
            mode = mode.strip().lower()
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float, str)):
                raise GuidanceDeclarationError(f"{where}: scale {scale!r} is not a number")
            try:
                scale = _scale(float(scale), where)
            except ValueError:
                raise GuidanceDeclarationError(f"{where}: scale {scale!r} is not a number") from None
        if mode in _NO_NEGATIVE_BRANCH and scale is not None and scale > 1.0:
            raise GuidanceDeclarationError(
                f"{where}: mode {mode!r} discards the negative branch, but scale {scale:g} > 1 would "
                f"turn its CFG combine on. Declare {{'mode': 'cfg', 'scale': {scale:g}}} if a guided "
                f"stream is intended, or drop the scale.")
        form = "mode+scale" if (mode is not None and scale is not None) else (
            "mode" if scale is None else "scale")
        return DeclaredGuidance(mode=mode, scale=scale, form=form)
    raise GuidanceDeclarationError(f"{where}: {type(value).__name__} is not a guidance declaration; "
                                   f"accepted: {ACCEPTED_FORMS}")


def _scale(value: float, where: str) -> float:
    number = float(value)
    if number != number or number < 0.0:  # NaN or negative
        raise GuidanceDeclarationError(f"{where}: guidance scale must be a non-negative number, got {value!r}")
    return number


def validate_declared_guidance(guidance: Mapping[str, Any] | None, *, where: str = "execution.guidance") -> None:
    """Every stream's value parses. Called at declaration load so a bad block fails at the boundary."""
    for stream, value in dict(guidance or {}).items():
        if not isinstance(stream, str) or not stream:
            raise GuidanceDeclarationError(f"{where}: stream names must be non-empty strings")
        parse_declared(value, where=f"{where}.{stream}")


@dataclass(frozen=True)
class ResolvedGuidance:
    """One stream's guidance as it will be SERVED: mode and scale, with where the scale came from."""

    stream: str
    mode: str
    #: the served scale; None only when the declaration named a mode and no family default is
    #: known to the caller (declaration-only tooling without an adapter)
    scale: float | None
    #: "declared" | "inherited from the family default" | "implied by mode" |
    #: "family default (not resolved here)"
    scale_source: str

    @property
    def negative_branch(self) -> bool:
        """Does serving this stream compute AND combine a negative branch? Scale 1 (or a
        positive-only / none mode) means no -- batch-1 for this stream's forwards."""
        return self.mode == "cfg" and (self.scale is None or self.scale > 1.0)

    @property
    def scale_inherited(self) -> bool:
        return self.scale_source != "declared"

    def canonical(self) -> dict[str, Any]:
        """The comparable form: mode + served scale. Two declarations that serve identically
        canonicalise identically ("positive_only" == {"mode": "positive_only", "scale": 1.0})."""
        return {"mode": self.mode, "scale": self.scale}

    def describe(self) -> str:
        scale = "?" if self.scale is None else f"{self.scale:g}"
        tail = "" if self.scale_source == "declared" else f" [{self.scale_source}]"
        return f"{self.stream}={self.mode}@{scale}{tail}"


def resolve(
    declared: Mapping[str, Any] | None,
    family: Mapping[str, tuple[str, float]] | None = None,
    *,
    where: str = "execution.guidance",
) -> dict[str, ResolvedGuidance]:
    """Resolve a declaration against the family's own (mode, scale) per stream.

    `family` is what the adapter states (`AdapterSpec.guidance`: mode + the model's deployed
    scale). Streams the family declares but the declaration omits are served at the family's
    values and recorded as inherited; streams the declaration names but the family does not
    declare are resolved from the declaration alone (an adapter that models fewer streams than
    a checkpoint declares is not an error for serving -- the serving path ignores them).
    """
    declared = dict(declared or {})
    family = dict(family or {})
    out: dict[str, ResolvedGuidance] = {}
    for stream in list(family) + [s for s in declared if s not in family]:
        fam = family.get(stream)
        fam_mode, fam_scale = (fam if fam is not None else (None, None))
        if stream not in declared:
            if fam_mode is None:
                continue
            scale = 1.0 if fam_mode in _NO_NEGATIVE_BRANCH else fam_scale
            out[stream] = ResolvedGuidance(stream, fam_mode, scale, "inherited from the family default")
            continue
        d = parse_declared(declared[stream], where=f"{where}.{stream}")
        if d.mode is None:
            # a bare scale is a CFG statement; <= 1 keeps the mode and turns the negative branch off
            mode = "cfg"
        else:
            mode = d.mode
        if d.scale is not None:
            scale, source = d.scale, "declared"
        elif mode in _NO_NEGATIVE_BRANCH:
            scale, source = 1.0, "implied by mode"
        elif fam_mode == "cfg" and fam_scale is not None:
            scale, source = float(fam_scale), "inherited from the family default"
        else:
            scale, source = None, "family default (not resolved here)"
        out[stream] = ResolvedGuidance(stream, mode, scale, source)
    return out


def canonical_guidance(declared: Mapping[str, Any] | None,
                       family: Mapping[str, tuple[str, float]] | None = None) -> dict[str, dict[str, Any]]:
    """{stream: {mode, scale}} -- the form two operating points are COMPARED on."""
    return {s: r.canonical() for s, r in sorted(resolve(declared, family).items())}


def capability_token(stream: str, value: Any) -> str:
    """The `guidance:<stream>=<...>` capability token for one declared value.

    The mode-string form keeps its historical token (`guidance:video=cfg`); a scale-bearing
    form adds the scale (`guidance:video=cfg@3`) so a plan can tell w=3 from the shipped w=5.
    """
    d = parse_declared(value, where=f"guidance.{stream}")
    if d.form == "mode":
        return f"guidance:{stream}={d.mode}"
    mode = d.mode or "cfg"
    return f"guidance:{stream}={mode}@{d.scale:g}"


def worker_flag_value(resolved: ResolvedGuidance) -> str:
    """How a resolved stream travels to a worker process as a `stream=value` CLI item.

    The worker re-parses through `parse_declared`, so the value must round-trip: a declared or
    implied scale travels as the number (the served w), a mode with an inherited-unknown scale
    travels as the mode name (the worker's own config keeps the family scale).
    """
    if resolved.mode in _NO_NEGATIVE_BRANCH:
        return resolved.mode
    if resolved.scale_source == "declared" and resolved.scale is not None:
        return f"{resolved.scale:g}"
    return resolved.mode


def describe_guidance(resolved: Mapping[str, ResolvedGuidance]) -> str:
    return ", ".join(r.describe() for _, r in sorted(resolved.items())) or "(none declared)"
