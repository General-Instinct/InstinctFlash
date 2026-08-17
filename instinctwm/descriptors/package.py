"""The published form of a checkpoint: directory layout, validation, and `from_pretrained`.

`checkpoint.py` answers *what a checkpoint declares*. This module answers *what a checkpoint IS on
disk*, so that a third party can publish one to the Hugging Face Hub and this runtime can serve it
without either side knowing anything about the other's training code.

THE LAYOUT

    my-checkpoint/
      instinctwm.json          REQUIRED. The declaration. Two namespaces: execution, provenance.
      config.json              REQUIRED. Backbone architecture, as the modelling library expects it.
      model.safetensors        REQUIRED (or a sharded set + model.safetensors.index.json).
      README.md                optional, strongly encouraged. The model card.
      tokenizer/ ...           optional, whatever the backbone needs.

Nothing else is required, and in particular nothing about training is required. A checkpoint is
publishable with `provenance` reduced to `{}` -- see `publishability()`. That is the property that
makes the platform claim true from the author's side: you can ship weights that this runtime serves
without shipping your recipe, your dataset, your teacher, or your loss curves.

WHY A SEPARATE MODULE. `load_declaration()` deliberately reads one file and returns one dataclass. It
should not also know about safetensors shards, Hub repo ids, or what a model card is. Packaging is a
different concern with a different failure mode: `load_declaration` fails when a declaration is wrong,
this fails when a *directory* is not a checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from instinctwm.descriptors.checkpoint import (
    FORBIDDEN_IN_EXECUTION, SCHEMA_VERSION, ExecutionDeclaration, load_declaration,
)

#: Read by the runtime. Every one of these must be present for `from_pretrained` to succeed.
REQUIRED = ("instinctwm.json", "config.json")

#: One of these must be present. A sharded checkpoint declares its shards in the index.
#:
#: The diffusers-named INDEX was missing from this list until the first real export tripped over it:
#: a sharded diffusers checkpoint has `diffusion_pytorch_model-0000N-of-0000M.safetensors` plus
#: `diffusion_pytorch_model.safetensors.index.json`, and only the unsharded name was listed. The
#: docstring already promised "or a sharded set + index"; the enumeration just did not match it.
WEIGHTS_ANY = (
    "model.safetensors",
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
)

#: Encouraged, never required, never read by the runtime.
OPTIONAL = ("README.md", "LICENSE", "tokenizer", "scheduler", "vae")

#: The smallest `execution` block that can be served. Everything else has a defensible default.
MINIMAL_EXECUTION = ("model_id", "backbone", "servable")


@dataclass(frozen=True)
class PackageReport:
    """What a directory is, and what it is missing. Never raises; the caller decides."""

    path: str
    is_checkpoint: bool
    missing: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    declaration: ExecutionDeclaration | None = None

    @property
    def ok(self) -> bool:
        return self.is_checkpoint and not self.missing and not self.problems

    def explain(self) -> str:
        out = [f"{self.path}", f"  servable package: {'YES' if self.ok else 'NO'}"]
        if self.declaration is not None:
            d = self.declaration
            out.append(f"  model_id={d.model_id!r} backbone={d.backbone!r} servable={d.servable} "
                       f"{'[LEGACY delta.json]' if d.legacy else ''}")
            if d.output_projection is not None:
                p = d.output_projection
                out.append(f"  output_projection: {p.kind} "
                           f"n_intervals={p.n_intervals} block={p.block} "
                           f"convention={p.velocity_convention}")
        for m in self.missing:
            out.append(f"  MISSING  {m}")
        for p in self.problems:
            out.append(f"  PROBLEM  {p}")
        for n in self.notes:
            out.append(f"  note     {n}")
        return "\n".join(out)


def validate_package(ckpt_dir: str | Path) -> PackageReport:
    """Check a directory against the published layout. Reports everything, raises nothing."""
    d = Path(ckpt_dir)
    missing: list[str] = []
    problems: list[str] = []
    notes: list[str] = []

    if not d.is_dir():
        return PackageReport(str(d), False, problems=(f"{d} is not a directory",))

    has_new, has_old = (d / "instinctwm.json").exists(), (d / "delta.json").exists()
    if not has_new and not has_old:
        return PackageReport(str(d), False, missing=("instinctwm.json",),
                             problems=("no declaration: this directory is not a checkpoint",))
    if not has_new and has_old:
        notes.append("legacy delta.json. Supported as a compatibility layer; publish instinctwm.json "
                     "for anything new. See `migrate_legacy()`.")

    for f in REQUIRED:
        if f == "instinctwm.json" and has_old and not has_new:
            continue
        if not (d / f).exists():
            missing.append(f)
    # WEIGHTS MAY BE SUPPLIED BY REFERENCE, and until this existed they could not be. `base_weights`
    # was already an execution fact -- LingBot-VA uses it to point at a frozen VAE/text-encoder stack
    # it does not vendor -- but the validator still demanded local weight files, so a declaration
    # could not adopt an upstream checkpoint wholesale. Found by declaring a LeRobot ACT policy:
    # every byte lives in `lerobot/act_...`, and the only sane package is a declaration plus a
    # pointer. Requiring a copy of somebody else's gigabytes to describe them is not a contract, it
    # is a tax. A package with neither local weights nor a pointer is still incomplete.
    has_local = any((d / w).exists() for w in WEIGHTS_ANY)
    pointer = None
    try:
        pointer = (load_declaration(d).extra or {}).get("base_weights")
    except Exception:                                            # reported below by the real load
        pass
    if not has_local and not pointer:
        missing.append(f"weights -- one of {', '.join(WEIGHTS_ANY)}, "
                       f"or an execution.base_weights pointer")
    elif not has_local:
        notes.append(f"no local weight files; they are referenced by "
                     f"execution.base_weights = {pointer!r}. The adapter resolves it at load, so "
                     f"that repo must stay reachable.")
    if not (d / "README.md").exists():
        notes.append("no README.md. Not required, but a published checkpoint without a model card is "
                     "hard to adopt.")

    decl = None
    try:
        decl = load_declaration(d)
    except Exception as e:                                   # the declaration is the whole contract
        problems.append(f"declaration rejected: {e}")
        return PackageReport(str(d), True, tuple(missing), tuple(problems), tuple(notes))

    if not decl.servable:
        notes.append("execution.servable is false, so the runtime will refuse to serve it. That is a "
                     "valid thing to publish -- weights people can fine-tune but not serve as-is.")
    for k in MINIMAL_EXECUTION:
        if not getattr(decl, k, None) and k != "servable":
            problems.append(f"execution.{k} is empty; it is part of the minimal serving metadata")

    return PackageReport(str(d), True, tuple(missing), tuple(problems), tuple(notes), decl)


def publishability(ckpt_dir: str | Path) -> tuple[bool, list[str]]:
    """Can this be published without exposing training internals?

    Returns (publishable, findings). Publishable means: the runtime can serve it with the
    `provenance` block removed entirely. If that is not true, something the runtime needs is
    currently living in the wrong namespace, which is the failure this whole two-namespace design
    exists to prevent.
    """
    d = Path(ckpt_dir)
    findings: list[str] = []
    p = d / "instinctwm.json"
    if not p.exists():
        return False, [f"{d}: no instinctwm.json. The legacy delta.json format has one flat namespace, "
                       f"so it cannot be published without also publishing training keys."]
    doc = json.loads(p.read_text())
    ex = dict(doc.get("execution") or {})

    leaked = sorted(k for k in FORBIDDEN_IN_EXECUTION if k in ex)
    if leaked:
        findings.append(f"execution block carries provenance keys {leaked} -- move them to provenance")

    # the real test: strip provenance and see whether it still loads and still declares enough
    stripped = {"instinctwm_schema": doc.get("instinctwm_schema", SCHEMA_VERSION), "execution": ex}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "instinctwm.json").write_text(json.dumps(stripped))
        try:
            decl = load_declaration(td)
        except Exception as e:
            findings.append(f"with provenance removed the declaration no longer loads: {e}")
            return False, findings
    if not decl.servable:
        findings.append("servable=false with provenance removed; the runtime would refuse it")

    prov = dict(doc.get("provenance") or {})
    if prov:
        findings.append(f"provenance present with {len(prov)} keys -- it will be published unless you "
                        f"remove it. The runtime never reads it either way.")
    return not any(f.startswith(("execution block carries", "with provenance removed")) for f in findings), findings


def migrate_legacy(ckpt_dir: str | Path, *, write: bool = False) -> dict:
    """Produce the `instinctwm.json` equivalent of a legacy `delta.json`.

    The execution block is what `_from_legacy_delta` already derives; everything else in the flat
    file becomes provenance, because the legacy format could not tell the difference and guessing
    the other way round would move a training key into the namespace the runtime reads.
    """
    from instinctwm.descriptors.checkpoint import provenance_of
    d = Path(ckpt_dir)
    decl = load_declaration(d)
    ex: dict = {"model_id": decl.model_id, "backbone": decl.backbone, "servable": decl.servable}
    if decl.guidance:
        ex["guidance"] = dict(decl.guidance)
    if decl.nfe:
        ex["nfe"] = dict(decl.nfe)
    if decl.output_projection is not None:
        p = decl.output_projection
        ex["output_projection"] = {"kind": p.kind, "n_intervals": p.n_intervals, "block": p.block,
                                   "velocity_convention": p.velocity_convention,
                                   "foldable": p.foldable}
    doc = {"instinctwm_schema": SCHEMA_VERSION, "execution": ex, "provenance": provenance_of(d)}
    if write:
        (d / "instinctwm.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc


@dataclass(frozen=True)
class Checkpoint:
    """A validated, servable checkpoint. Holds execution facts and a path; never provenance."""

    path: str
    execution: ExecutionDeclaration
    report: PackageReport = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def model_id(self) -> str:
        return self.execution.model_id

    def capabilities(self) -> frozenset[str]:
        """The capability tokens the runtime may plan against.

        Derived ONLY from the execution block. A planner that wants to know whether some optimization
        applies asks this; it never asks how the checkpoint was trained, because there is nothing here
        that could answer.
        """
        caps = set()
        if self.execution.servable:
            caps.add("servable")
        if self.execution.backbone:
            caps.add(f"backbone:{self.execution.backbone}")
        p = self.execution.output_projection
        if p is not None and p.kind:
            caps.add(f"output_projection:{p.kind}")
            if p.foldable:
                caps.add("output_projection:foldable")
        for k, v in (self.execution.guidance or {}).items():
            caps.add(f"guidance:{k}={v}")
        for k in (self.execution.extra or {}):
            caps.add(f"declares:{k}")
        return frozenset(caps)


def from_pretrained(model_id_or_path: str | Path, *, revision: str | None = None,
                    require_servable: bool = True) -> Checkpoint:
    """Load a checkpoint by local path, or by Hub repo id if `huggingface_hub` is installed.

    Reads the EXECUTION block only. There is no argument that selects a training method, and no code
    path in this function that could branch on one -- `load_declaration` does not return provenance,
    so there is nothing to branch on.

    `require_servable=False` is for tooling that wants to inspect a checkpoint it will not serve.
    """
    p = Path(model_id_or_path)
    if not p.exists():
        try:
            from huggingface_hub import snapshot_download        # optional dependency
        except ImportError as e:
            raise RuntimeError(
                f"{model_id_or_path!r} is not a local directory and huggingface_hub is not installed, "
                f"so it cannot be resolved as a Hub repo id. Install huggingface_hub, or pass a path."
            ) from e
        p = Path(snapshot_download(str(model_id_or_path), revision=revision))

    report = validate_package(p)
    if not report.is_checkpoint or report.declaration is None:
        raise RuntimeError(f"{p} is not a servable checkpoint:\n{report.explain()}")
    if report.missing or report.problems:
        raise RuntimeError(f"{p} is incomplete:\n{report.explain()}")
    if require_servable:
        report.declaration.require_servable(f"from_pretrained({model_id_or_path!r})")
    return Checkpoint(str(p), report.declaration, report)


if __name__ == "__main__":                                   # python -m instinctwm.descriptors.package
    import sys as _s
    if len(_s.argv) != 2:
        print("usage: python -m instinctwm.descriptors.package <checkpoint-dir>")
        raise SystemExit(2)
    _rep = validate_package(_s.argv[1])
    print(_rep.explain())
    _ok, _find = (False, ["not an instinctwm.json package"])
    try:
        _ok, _find = publishability(_s.argv[1])
    except Exception as _e:                                  # noqa: BLE001
        _find = [str(_e)]
    print(f"  publishable without training internals: {'YES' if _ok else 'NO'}")
    for _f in _find:
        print(f"    - {_f}")
    raise SystemExit(0 if _rep.ok else 1)
