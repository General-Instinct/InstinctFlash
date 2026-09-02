"""Reading a checkpoint's declaration, and refusing to hand the runtime anything it must not read.

TWO NAMESPACES, ENFORCED BY THE READER

`instinctflash.json` has exactly two top-level blocks:

    execution     what the runtime may read. Capabilities and structure.
    provenance    training method, recipe, dataset, optimizer, diagnostics, certification.
                  FOR HUMANS. `load_declaration()` does not return it.

The enforcement is the point. The previous format, `delta.json`, was one flat dict containing both
`n_intervals` (execution) and `coverage_gate_pass` (a PDD training statistic) -- so the serving path
reading a training key was a one-line mistake rather than a boundary violation, and it duly happened.
See AUDIT.md F2 and F4.

`ExecutionDeclaration` therefore has no field that could carry a recipe. `provenance` is parsed only to
be dropped, and `provenance_of()` exists separately for tools that legitimately want it (model cards,
reproduction, `verify/`). Nothing under `runtime/` calls it.

THE `servable` FIELD replaces the recipe-specific gate. The runtime asks one recipe-agnostic question:
is this checkpoint fit to serve? Whoever publishes it answers. WHY it is or is not fit -- coverage
gates, endpoint RMSE, head update counts -- lives in provenance, where the runtime cannot reach it.

LEGACY. `delta.json` is still read when `instinctflash.json` is absent, so existing checkpoints under
/home/ubuntu/iwm_results/ keep serving unchanged. The legacy path maps `coverage_gate_pass ->
servable`, which is the ONE place that key is still consulted, and it is quarantined in
`_from_legacy_delta` with a deprecation note rather than sitting in the serving path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1

#: Keys that must never appear in an `execution` block. Checked on load, so a mis-stamped checkpoint
#: fails loudly at the boundary instead of quietly teaching the runtime to read provenance.
FORBIDDEN_IN_EXECUTION = (
    "recipe", "training_method", "teacher", "student", "solver", "dataset", "optimizer",
    "coverage_gate_pass", "min_updates_per_head", "head_updates_min", "endpoint_rmse",
    "trainable", "paper", "training_diagnostics", "certification",
)


@dataclass(frozen=True)
class OutputProjection:
    """What the checkpoint's final projection provides. A CAPABILITY, not a recipe.

    `kind == "per_interval_velocity_heads"` is what replaces "this is a PDD checkpoint". Any recipe --
    DMD2, LCM, or one not yet written -- producing L linear heads per block over an N-interval grid
    declares the same three numbers and is served by the same code.
    """

    kind: str
    n_intervals: int | None = None
    block: int | None = None
    #: "sigma_descending" or "t_ascending". A double sign flip here once produced 0/100 on RoboTwin
    #: against a 92/100 control, diagnosed only by reading the training loss's convention. A comment
    #: cannot be checked; a field can.
    velocity_convention: str = "sigma_descending"
    foldable: bool = True

    PER_INTERVAL_VELOCITY_HEADS = "per_interval_velocity_heads"

    def nfe(self) -> int:
        if not self.n_intervals or not self.block:
            raise ValueError(f"{self.kind}: n_intervals and block are required to derive NFE")
        return self.n_intervals // self.block


@dataclass(frozen=True)
class ExecutionDeclaration:
    """Everything the runtime may read. Deliberately has nowhere to put a training method."""

    model_id: str = ""
    backbone: str = ""
    servable: bool = False
    guidance: Mapping[str, Any] = field(default_factory=dict)
    nfe: Mapping[str, int] = field(default_factory=dict)
    output_projection: OutputProjection | None = None
    #: Anything else declared under `execution`, minus the forbidden keys. Kept so a new capability can
    #: be declared before this dataclass grows a field for it.
    extra: Mapping[str, Any] = field(default_factory=dict)
    #: Provenance-only for humans: which file this came from, and whether it was the legacy format.
    source: str = ""
    legacy: bool = False

    def require_servable(self, where: str) -> None:
        """Refuse an unservable checkpoint, without asking why it is unservable.

        The reason is a training fact and lives in provenance. This check is the recipe-agnostic
        successor to reading `coverage_gate_pass` in the serving path.
        """
        if not self.servable:
            raise RuntimeError(
                f"{where}: the checkpoint declares servable=false, so it is not fit to serve and a "
                f"number measured from it could not be defended. The reason is a training fact and "
                f"lives in the provenance block"
                + (" (legacy delta.json: coverage_gate_pass was false)." if self.legacy else "."))

    def require_projection(self, kind: str, where: str) -> OutputProjection:
        p = self.output_projection
        if p is None or p.kind != kind:
            raise RuntimeError(
                f"{where}: needs output_projection.kind == {kind!r}, checkpoint declares "
                f"{(p.kind if p else None)!r}. The serving path is chosen by CAPABILITY, so a "
                f"checkpoint that does not declare this one is not servable here regardless of how it "
                f"was trained.")
        return p


def _from_legacy_delta(meta: Mapping[str, Any], source: str) -> ExecutionDeclaration:
    """Map the old flat `delta.json` onto the two-namespace model.

    QUARANTINE. This is the only function in the package that reads `coverage_gate_pass`, and it reads
    it to produce `servable` -- a recipe-agnostic boolean -- so that no caller has to. `recipe`,
    `solver`, `endpoint_rmse` and friends are present in the same dict and are deliberately ignored.
    """
    from instinctflash.descriptors.guidance import validate_declared_guidance

    validate_declared_guidance(dict(meta.get("guidance", {})), where=f"{source}: guidance")
    op = None
    if "n_intervals" in meta and "block" in meta:
        op = OutputProjection(
            kind=OutputProjection.PER_INTERVAL_VELOCITY_HEADS,
            n_intervals=int(meta["n_intervals"]), block=int(meta["block"]),
            # The legacy format never declared this. Every checkpoint written under it trained through
            # the adapter that negates once, so the head's raw output is already the sigma-velocity.
            velocity_convention="sigma_descending", foldable=True)
    return ExecutionDeclaration(
        model_id=str(meta.get("model_id", "")),
        backbone=str(meta.get("backbone", "")),
        servable=bool(meta.get("coverage_gate_pass", False)),
        guidance=dict(meta.get("guidance", {})),
        nfe=dict(meta.get("nfe", {})),
        output_projection=op,
        source=source, legacy=True,
    )


#: accepted declaration filenames, preferred first. The second is the pre-rename name; read it
#: forever, write only the first.
DECLARATION_FILENAMES = ("instinctflash.json", "instinctwm.json")


def _declaration_file(d: Path) -> "Path | None":
    for name in DECLARATION_FILENAMES:
        if (d / name).exists():
            return d / name
    return None


def load_declaration(ckpt_dir: str | Path) -> ExecutionDeclaration:
    """Read a checkpoint's EXECUTION declaration. Never returns provenance.

    Resolution order, first hit wins:
      1. `instinctflash.json`  -- the two-namespace schema
      2. `instinctwm.json`     -- the same schema under the project's pre-rename name. Published
                                  artifacts may carry it, and a rename must not orphan them.
      3. `delta.json`          -- legacy flat format, mapped and marked
      4. refuse                -- an unrecognised checkpoint is not served on guessed facts
    """
    d = Path(ckpt_dir)
    new, old = _declaration_file(d), d / "delta.json"

    if new is not None and new.exists():
        doc = json.loads(new.read_text())
        # the schema KEY was also renamed; pre-rename documents say instinctwm_schema
        ver = int(doc.get("instinctflash_schema", doc.get("instinctwm_schema", 0)))
        if ver != SCHEMA_VERSION:
            raise RuntimeError(f"{new}: instinctflash_schema {ver}, this runtime speaks {SCHEMA_VERSION}")
        ex = dict(doc.get("execution") or {})
        if not ex:
            raise RuntimeError(f"{new}: no `execution` block. Execution facts and provenance are "
                               f"separate namespaces; a declaration with only provenance is not "
                               f"servable.")
        leaked = sorted(k for k in FORBIDDEN_IN_EXECUTION if k in ex)
        if leaked:
            raise RuntimeError(
                f"{new}: provenance keys {leaked} appear in the `execution` block. These describe how "
                f"the checkpoint was TRAINED and the runtime must not read them; move them under "
                f"`provenance`. See CHECKPOINTS.md.")
        # The guidance block is checked here, at the boundary: per stream a mode name, a numeric
        # scale, or {mode, scale} (descriptors/guidance.py). A value the runtime could not serve
        # as written is refused now, not ignored at serve time.
        from instinctflash.descriptors.guidance import GuidanceDeclarationError, validate_declared_guidance
        try:
            validate_declared_guidance(dict(ex.get("guidance") or {}), where=f"{new}: execution.guidance")
        except GuidanceDeclarationError as e:
            raise RuntimeError(str(e)) from None
        opd = dict(ex.pop("output_projection", {}) or {})
        op = OutputProjection(
            kind=str(opd.get("kind", "")),
            n_intervals=(int(opd["n_intervals"]) if "n_intervals" in opd else None),
            block=(int(opd["block"]) if "block" in opd else None),
            velocity_convention=str(opd.get("velocity_convention", "sigma_descending")),
            foldable=bool(opd.get("foldable", True)),
        ) if opd else None
        return ExecutionDeclaration(
            model_id=str(ex.pop("model_id", "")), backbone=str(ex.pop("backbone", "")),
            servable=bool(ex.pop("servable", False)),
            guidance=dict(ex.pop("guidance", {})), nfe=dict(ex.pop("nfe", {})),
            output_projection=op, extra=ex, source=str(new), legacy=False,
        )

    if old.exists():
        return _from_legacy_delta(json.loads(old.read_text()), str(old))

    raise RuntimeError(
        f"{d}: no instinctflash.json and no delta.json. A checkpoint must declare what it needs in order "
        f"to run; it is not served on guessed facts.")


def provenance_of(ckpt_dir: str | Path) -> dict:
    """The provenance block, FOR HUMANS AND TOOLS. Never called from the runtime path.

    Kept in this module so that "where does provenance live" has one answer, and separate from
    `load_declaration` so that reaching it is a deliberate act rather than a dictionary lookup away
    from the execution facts.
    """
    d = Path(ckpt_dir)
    new = d / "instinctflash.json"
    if new.exists():
        return dict(json.loads(new.read_text()).get("provenance") or {})
    old = d / "delta.json"
    if old.exists():
        # In the legacy format everything shares one namespace, so provenance is what is left after
        # the execution keys are removed.
        meta = json.loads(old.read_text())
        exec_keys = {"model_id", "backbone", "guidance", "nfe", "n_intervals", "block",
                     "shapes", "dtypes"}
        return {k: v for k, v in meta.items() if k not in exec_keys}
    return {}
