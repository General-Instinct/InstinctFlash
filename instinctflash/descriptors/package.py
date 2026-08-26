"""The published form of a checkpoint: directory layout, validation, and `from_pretrained`.

`checkpoint.py` answers *what a checkpoint declares*. This module answers *what a checkpoint IS on
disk*, so that a third party can publish one to the Hugging Face Hub and this runtime can serve it
without either side knowing anything about the other's training code.

THE LAYOUT

    my-checkpoint/
      instinctflash.json          REQUIRED. The declaration. Two namespaces: execution, provenance.
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

from instinctflash.descriptors.checkpoint import (
    FORBIDDEN_IN_EXECUTION, SCHEMA_VERSION, ExecutionDeclaration, load_declaration,
)

#: Read by the runtime. Every one of these must be present for `from_pretrained` to succeed.
REQUIRED = ("instinctflash.json", "config.json")

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


def _training_layout_hint(d: Path) -> "str | None":
    """One actionable line when a FAILING directory is a training-output tree.

    Trainers write `<run>/transformer/*.safetensors`; the package convention is the transformer
    contents FLAT at the package root, next to the declaration (the layout `materialize()`
    composes from — see adapters/lingbot_va.py). A user pointing `validate` at the training
    output gets "MISSING config.json" and "no local weight files; referenced by base_weights",
    both true and both useless without this line, because the weights are sitting one directory
    down. Only emitted when validation is already failing: a composed upstream package
    legitimately carries a transformer/ component and must not be told to flatten itself.
    """
    t = d / "transformer"
    if not t.is_dir() or not any(t.glob("*.safetensors")):
        return None
    return ("this looks like a training-output layout: the weight files sit under transformer/. "
            "An InstinctFlash package is the FLAT transformer contents at the package root, next "
            f"to instinctflash.json — run  mv {t}/* {d}/  (config.json and the *.safetensors land "
            "at the root); the frozen vae/text_encoder/tokenizer are never packaged, they come "
            "from the execution.base_weights pointer")


def validate_package(ckpt_dir: str | Path) -> PackageReport:
    """Check a directory against the published layout. Reports everything, raises nothing."""
    d = Path(ckpt_dir)
    missing: list[str] = []
    problems: list[str] = []
    notes: list[str] = []

    if not d.is_dir():
        return PackageReport(str(d), False, problems=(f"{d} is not a directory",))

    from instinctflash.descriptors.checkpoint import _declaration_file
    has_new, has_old = _declaration_file(d) is not None, (d / "delta.json").exists()
    if not has_new and not has_old:
        hint = _training_layout_hint(d)
        return PackageReport(str(d), False, missing=("instinctflash.json",),
                             problems=("no declaration: this directory is not a checkpoint",),
                             notes=(hint,) if hint else ())
    if not has_new and has_old:
        notes.append("legacy delta.json. Supported as a compatibility layer; publish instinctflash.json "
                     "for anything new. See `migrate_legacy()`.")

    for f in REQUIRED:
        if f == "instinctflash.json":
            if has_new or has_old:
                continue                       # any accepted declaration name satisfies it
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

    def finished_notes() -> tuple[str, ...]:
        # The layout hint rides on FAILING reports only, appended last so it reads as the fix.
        hint = _training_layout_hint(d) if (missing or problems) else None
        return tuple([*notes, hint] if hint else notes)

    decl = None
    try:
        decl = load_declaration(d)
    except Exception as e:                                   # the declaration is the whole contract
        problems.append(f"declaration rejected: {e}")
        return PackageReport(str(d), True, tuple(missing), tuple(problems), finished_notes())

    if not decl.servable:
        notes.append("execution.servable is false, so the runtime will refuse to serve it. That is a "
                     "valid thing to publish -- weights people can fine-tune but not serve as-is.")
    for k in MINIMAL_EXECUTION:
        if not getattr(decl, k, None) and k != "servable":
            problems.append(f"execution.{k} is empty; it is part of the minimal serving metadata")

    return PackageReport(str(d), True, tuple(missing), tuple(problems), finished_notes(), decl)


def verify_weights_indexes(ckpt_dir: str | Path) -> list[str]:
    """Integrity of a sharded-weights index: every declared shard exists, inside the package.

    Three failure classes, each of which `validate_package` used to wave through because it reads
    declarations, not weights: a weight_map that is missing or empty, a referenced shard that does
    not exist, and a shard path that escapes the package directory (a symlink or ../ that would
    make a 'validated' package read files it does not contain). Returns problems; empty is good.
    A package with no index file (single-file weights, pointer-only) has nothing to verify.
    """
    root = Path(ckpt_dir)
    problems: list[str] = []
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        p = root / name
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text())
            weight_map = doc.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                problems.append(f"{name}: missing non-empty weight_map")
                continue
            for shard in sorted(set(weight_map.values())):
                target = root / str(shard)
                try:
                    target.resolve().relative_to(root.resolve())
                except ValueError:
                    problems.append(f"{name}: shard escapes package: {shard}")
                    continue
                if not target.is_file():
                    problems.append(f"{name}: referenced shard is missing: {shard}")
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{name}: invalid index: {type(e).__name__}: {e}")
    return problems


def publishability(ckpt_dir: str | Path) -> tuple[bool, list[str]]:
    """Can this be published without exposing training internals?

    Returns (publishable, findings). Publishable means: the runtime can serve it with the
    `provenance` block removed entirely. If that is not true, something the runtime needs is
    currently living in the wrong namespace, which is the failure this whole two-namespace design
    exists to prevent.
    """
    d = Path(ckpt_dir)
    findings: list[str] = []
    from instinctflash.descriptors.checkpoint import _declaration_file
    p = _declaration_file(d)
    if p is None:
        return False, [f"{d}: no instinctflash.json. The legacy delta.json format has one flat namespace, "
                       f"so it cannot be published without also publishing training keys."]
    doc = json.loads(p.read_text())
    ex = dict(doc.get("execution") or {})

    leaked = sorted(k for k in FORBIDDEN_IN_EXECUTION if k in ex)
    if leaked:
        findings.append(f"execution block carries provenance keys {leaked} -- move them to provenance")

    # the real test: strip provenance and see whether it still loads and still declares enough
    stripped = {"instinctflash_schema": doc.get("instinctflash_schema", SCHEMA_VERSION), "execution": ex}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "instinctflash.json").write_text(json.dumps(stripped))
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
    """Produce the `instinctflash.json` equivalent of a legacy `delta.json`.

    The execution block is what `_from_legacy_delta` already derives; everything else in the flat
    file becomes provenance, because the legacy format could not tell the difference and guessing
    the other way round would move a training key into the namespace the runtime reads.
    """
    from instinctflash.descriptors.checkpoint import provenance_of
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
    doc = {"instinctflash_schema": SCHEMA_VERSION, "execution": ex, "provenance": provenance_of(d)}
    if write:
        (d / "instinctflash.json").write_text(json.dumps(doc, indent=2) + "\n")
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


def _declared_view(snapshot: Path, model_id: str, doc: dict) -> Path:
    """A checkpoint directory for an upstream snapshot that carries no declaration.

    Symlinks every snapshot entry (weights stay in the HF cache, nothing is copied or mutated)
    and writes the known declaration next to them. Rebuilt on every call: cheap, and it tracks
    snapshot updates.
    """
    import json as _json
    import os
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    view = base / "instinctflash" / "declared" / model_id.replace("/", "__")
    view.mkdir(parents=True, exist_ok=True)
    for entry in view.iterdir():
        if entry.is_symlink() or entry.name == "instinctflash.json":
            entry.unlink()
    for entry in snapshot.iterdir():
        (view / entry.name).symlink_to(entry)
    if not (view / "config.json").exists() and (snapshot / "transformer" / "config.json").exists():
        # diffusers-composed upstream layout: the backbone architecture config lives under
        # transformer/; the package contract wants it at top level, and it is the same file
        (view / "config.json").symlink_to(snapshot / "transformer" / "config.json")
    (view / "instinctflash.json").write_text(_json.dumps(doc, indent=1))
    return view


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
        from instinctflash.descriptors.checkpoint import _declaration_file
        if _declaration_file(p) is None:
            # A known upstream release: its authors publish weights without a declaration, and we
            # serve them without republishing. Materialize a declared view — symlinks into the HF
            # snapshot plus our declaration — so every check below runs unchanged on it.
            from instinctflash.descriptors.known import lookup
            doc = lookup(str(model_id_or_path))
            if doc is not None:
                p = _declared_view(p, str(model_id_or_path), doc)

    report = validate_package(p)
    if not report.is_checkpoint or report.declaration is None:
        raise RuntimeError(f"{p} is not a servable checkpoint:\n{report.explain()}")
    if report.missing or report.problems:
        raise RuntimeError(f"{p} is incomplete:\n{report.explain()}")
    if require_servable:
        report.declaration.require_servable(f"from_pretrained({model_id_or_path!r})")
    return Checkpoint(str(p), report.declaration, report)


if __name__ == "__main__":                                   # python -m instinctflash.descriptors.package
    import sys as _s
    if len(_s.argv) != 2:
        print("usage: python -m instinctflash.descriptors.package <checkpoint-dir>")
        raise SystemExit(2)
    _rep = validate_package(_s.argv[1])
    print(_rep.explain())
    _ok, _find = (False, ["not an instinctflash.json package"])
    try:
        _ok, _find = publishability(_s.argv[1])
    except Exception as _e:                                  # noqa: BLE001
        _find = [str(_e)]
    print(f"  publishable without training internals: {'YES' if _ok else 'NO'}")
    for _f in _find:
        print(f"    - {_f}")
    raise SystemExit(0 if _rep.ok else 1)
