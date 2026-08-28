"""`instinctflash` — the command line. Two verbs, no Python required.

    instinctflash serve     <model-id>       deploy: preflight, then serve over websocket
    instinctflash validate  <dir>            trust: publishable-package check + the certificate

`serve` always prints its preflight — device capabilities, the checkpoint's declaration, and the
plan — BEFORE any weight is downloaded, because those are the questions you want answered before
committing to a download or a GPU. A local directory with NO declaration is scaffolded inline
first (the same writer as `validate --validate.scaffold=auto`, announced field by field): when the
checkpoint proves everything, serve continues in the same command; when FILL_ME facts remain, it
stops before any download with exactly the missing fields and the rerun. Its flags select how far
to go:

    --serve.dry_run=true    preflight only: no download, no GPU, exit
    --serve.smoke=true      load, produce one zero-filled action, exit — proves the checkpoint
                            loads here and returns finite actions; it is not an evaluation
    --serve.viz=true        stream observations, actions and latency to a Rerun viewer

`validate` is the structural package check; given `--validate.teacher_outcomes`,
`--validate.student_outcomes` and `--validate.margin` it also runs the paired non-inferiority
analysis and stamps the certificate into the package's provenance block, and a later plain
`validate <dir>` verifies any embedded certificate's integrity. A checkpoint with no
declaration yet starts with `--validate.scaffold=<base-hub-id|auto>`, which WRITES its
instinctflash.json from a built-in base declaration — inferring what the checkpoint itself
proves, marking the rest FILL_ME — and then validates the result, flagging every sentinel.

Both verbs use the typed dotted-field syntax from `cli_config` (`--serve.smoke=true`,
`--validate.margin=-0.05`, optional `--config_path=FILE` with CLI overrides winning, unknown
fields are hard errors, JSON errors use one stable schema, `--output.path` writes atomically).

The previous verbs — devices, describe, plan, run, certify — remain as undocumented compatibility
aliases: each prints a one-line pointer to the verb that absorbed it and then behaves exactly as
before, so existing scripts keep working.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from instinctflash.cli_config import OutputConfig, RuntimeConfig


@dataclass
class ServeOptions:
    """The `serve` verb's own knobs. Runtime knobs (device, placement, nfe, tier_ceiling,
    exclude_passes) are the shared `--runtime.*` section, so they are the same words here as
    everywhere else."""

    model: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    #: preflight only — device + declaration + plan, no download, no GPU
    dry_run: bool = False
    #: load, produce one zero-filled action, exit; a load check, never an evaluation
    smoke: bool = False
    #: stream observations/actions/latency to a Rerun viewer (the `viz` extra)
    viz: bool = False
    #: where the Rerun stream goes: "" spawns a viewer, "*.rrd" records headless,
    #: "rerun+http://..." connects to a running viewer
    viz_sink: str = ""
    #: seed the RNGs the model draws noise from, per episode (adapters that can thread it deeper
    #: seed per request — wan_va seeds every _infer draw). Two serves with the same seed and the
    #: same inputs are comparable value-for-value; unset keeps the model's own unseeded behaviour.
    seed: int | None = None


@dataclass
class ServeConfig:
    serve: ServeOptions = field(default_factory=ServeOptions)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class ValidateOptions:
    """The `validate` verb: the structural check, plus the certificate when outcomes are given."""

    path: Path | None = None
    #: write the package's instinctflash.json BEFORE validating it: a built-in base id (e.g.
    #: lerobot/pi05_base) to copy-and-infer from, or "auto" to detect the base from the
    #: checkpoint's own config.json. Facts the checkpoint does not prove are written as FILL_ME
    #: sentinels, and the validation that follows flags every one of them.
    scaffold: str = ""
    #: allow --validate.scaffold to overwrite an existing declaration file. Without it an
    #: existing instinctflash.json is never touched; the scaffold prints what would change instead.
    force: bool = False
    teacher_outcomes: Path | None = None
    student_outcomes: Path | None = None
    margin: float | None = None
    min_pairs: int = 1
    harness: str | None = None
    recipe: str | None = None
    seeds: list[int] | None = None
    per_task: bool = True


@dataclass
class ValidateConfig:
    validate: ValidateOptions = field(default_factory=ValidateOptions)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class CertifyOptions:
    """The `certify` verb's inputs. Module-level so the typed parser can resolve the hints."""

    teacher_outcomes: Path | None = None
    student_outcomes: Path | None = None
    margin: float | None = None
    min_pairs: int = 1
    harness: str | None = None
    recipe: str | None = None
    seeds: list[int] | None = None
    per_task: bool = True
    teacher_hash: str = "?"
    student_hash: str = "?"


@dataclass
class CertifyConfig:
    certify: CertifyOptions = field(default_factory=CertifyOptions)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(format="json"))


def _fmt_device(d) -> str:
    cap = f"sm{d.capability[0]}{d.capability[1]}" if d.capability != (0, 0) else "cpu"
    mem = f"{d.total_memory / 1e9:.0f} GB" if d.total_memory else "-"
    cls, why = d.device_class()
    return (f"{d.name}  {cap}  {mem}\n  features: {', '.join(sorted(d.features))}\n"
            f"  class   : {cls} — {why}")


def cmd_devices(a) -> int:
    from instinctflash.passes.contract import KNOWN_FEATURES, DeviceProfile
    try:
        d = DeviceProfile.probe()
    except Exception as e:                                       # noqa: BLE001
        # EXIT 0. "There is no accelerator here, and here is why" is a successful answer to the
        # question this verb asks, not a failure to answer it -- the core install has no torch by
        # design, so this is the expected state on a laptop rather than an error. It also mattered
        # concretely: exiting non-zero made `instinctflash devices` kill scripts/check_release.sh
        # through pipefail, so the verb that reports capability was breaking the release gate.
        print(f"no accelerator visible: {type(e).__name__}: {e}")
        print("This is expected without the `runtime` extra installed. Planning still works; passes "
              "with hardware requirements report APPLICABILITY UNCHECKED.")
        return 0
    print(_fmt_device(d))
    absent = sorted(KNOWN_FEATURES - d.features)
    if absent:
        print(f"  absent  : {', '.join(absent)}")
    print("\nPasses and backends declare requirements against exactly these names, so anything in "
          "'absent' will decline here and say so in the plan.")
    return 0


def cmd_describe(a) -> int:
    from instinctflash import describe
    try:
        d = describe(a.model)
    except Exception as e:                                       # noqa: BLE001
        print(f"{a.model}: {type(e).__name__}: {e}")
        return 1
    if a.json:
        print(json.dumps(d, indent=2))
        return 0
    for k in ("model_id", "backbone", "servable", "nfe", "guidance"):
        print(f"  {k:12} {d[k]}")
    print(f"  {'capabilities':12} {', '.join(d['capabilities'])}")
    if d.get("has_provenance"):
        print(f"  {'provenance':12} present, and never read by the runtime")
    return 0


def _certificate_content_hash(block: dict) -> str:
    """The self-hash a stamped certificate carries, over everything except the hash itself."""
    import hashlib

    payload = {k: v for k, v in block.items() if k != "content_sha256"}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def _declaration_path(pkg: Path) -> Path | None:
    from instinctflash.descriptors.checkpoint import DECLARATION_FILENAMES
    for name in DECLARATION_FILENAMES:
        if (pkg / name).is_file():
            return pkg / name
    return None


def _stamp_certificate(pkg: Path, v: "ValidateOptions") -> tuple[Any, dict]:
    """Run the paired analysis and stamp the result into the package's provenance block.

    Provenance is the one namespace the runtime never reads (descriptors enforce that), which is
    exactly why the certificate belongs there: it is a fact about how the checkpoint was verified,
    not an input to serving. The block carries sha256 of both outcome files and a self-hash, so a
    later plain `validate` can detect a hand-edited verdict.
    """
    import hashlib
    from datetime import datetime, timezone

    from instinctflash.cli_config import ConfigError, _atomic_write
    from instinctflash.verify.certify import certify, load_jsonl

    decl_path = _declaration_path(pkg)
    if decl_path is None:
        raise ConfigError(f"cannot stamp a certificate: {pkg} has no declaration file "
                          f"(instinctflash.json) to carry a provenance block")

    def sha256(p: Path) -> str:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    t_sha, s_sha = sha256(v.teacher_outcomes), sha256(v.student_outcomes)
    cert = certify(
        load_jsonl(str(v.teacher_outcomes)), load_jsonl(str(v.student_outcomes)),
        margin=v.margin, min_pairs=v.min_pairs,
        teacher_hash=t_sha, student_hash=s_sha,
        harness=v.harness or "?", recipe=v.recipe or "?",
        seeds=",".join(map(str, v.seeds)) if v.seeds is not None else "?",
    )
    block = json.loads(cert.to_json())
    block.update({
        "teacher_outcomes_sha256": t_sha,
        "student_outcomes_sha256": s_sha,
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instinctflash_version": _cli_version(),
    })
    block["content_sha256"] = _certificate_content_hash(block)

    doc = json.loads(decl_path.read_text())
    doc.setdefault("provenance", {})["certificate"] = block
    _atomic_write(decl_path, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return cert, block


def _verify_embedded_certificate(pkg: Path) -> tuple[str, dict | None]:
    """-> (status, block): 'absent', 'intact' or 'tampered'."""
    decl_path = _declaration_path(pkg)
    if decl_path is None:
        return "absent", None
    try:
        doc = json.loads(decl_path.read_text())
    except Exception:                                            # noqa: BLE001 - structure check reports it
        return "absent", None
    block = (doc.get("provenance") or {}).get("certificate")
    if not isinstance(block, dict):
        return "absent", None
    ok = block.get("content_sha256") == _certificate_content_hash(block)
    return ("intact" if ok else "tampered"), block


def _cli_version() -> str:
    from instinctflash.cli_config import _version
    return _version()


def cmd_validate(argv: list[str]) -> int:
    """`instinctflash validate <dir>` — the trust verb.

    Structure always; the certificate when outcomes are given (reusing `verify.certify`, the same
    code path the harnesses use, so a certificate produced here IS the certificate); integrity of
    any embedded certificate on every plain run.
    """
    from instinctflash.cli_config import CommandReport, ConfigError, execute

    if argv and not argv[0].startswith("-"):
        argv = [f"--validate.path={argv[0]}", *argv[1:]]

    def run(cfg: ValidateConfig) -> CommandReport:
        v = cfg.validate
        if v.path is None:
            raise ConfigError("validate.path is required: instinctflash validate <dir>")
        from instinctflash.descriptors.package import (
            publishability, validate_package, verify_weights_indexes,
        )
        lines: list[str] = []
        result: dict[str, Any] = {"path": str(v.path)}
        scaffold_ok = True
        if v.scaffold:
            # Write the declaration FIRST, then fall through to the normal validation below, so
            # the user sees the scaffold judged by exactly the check everyone else's package
            # gets — including a PROBLEM line for every FILL_ME the scaffold refused to guess.
            from instinctflash.descriptors.scaffold import ScaffoldError, run_scaffold
            try:
                sres, stext, wrote = run_scaffold(Path(v.path), v.scaffold, force=bool(v.force))
            except ScaffoldError as e:
                raise ConfigError(str(e)) from e
            lines += [stext, ""]
            result["scaffold"] = sres
            # Refusing to overwrite is correct behaviour AND a non-zero exit: the command was
            # asked to write a declaration and did not. The diff above is the answer.
            scaffold_ok = wrote
        rep = validate_package(str(v.path))
        lines.append(rep.explain())
        # Weights-index integrity GATES the exit code: a package whose declared shards are missing,
        # or whose shard paths escape the package, is not a valid package. Publishability stays
        # INFORMATIONAL (exit-neutral), preserving the published exit contract for packages that
        # are valid but carry training internals.
        index_problems = verify_weights_indexes(str(v.path))
        for p in index_problems:
            lines.append(f"  PROBLEM  {p}")
        # FILL_ME sentinels GATE the exit code, on every run: a scaffolded declaration is not a
        # valid package until its author has replaced the last fact the scaffold refused to
        # guess. Each line carries the scaffold's one-line explanation of what belongs there.
        from instinctflash.descriptors.scaffold import fill_me_findings
        fill_me = fill_me_findings(str(v.path))
        for where, why in fill_me:
            lines.append(f"  PROBLEM  {where} is \"FILL_ME\" — {why}")
        publishable = False
        try:
            publishable, findings = publishability(str(v.path))
            lines.append(f"  publishable without training internals: {'YES' if publishable else 'NO'}")
            for f in findings:
                lines.append(f"    - {f}")
        except Exception as e:                                   # noqa: BLE001
            lines.append(f"  publishability: {type(e).__name__}: {e}")

        result.update({"structure_ok": rep.ok,
                       "index_problems": list(index_problems),
                       "fill_me": [where for where, _ in fill_me],
                       "publishable": publishable})
        ok = rep.ok and not index_problems and not fill_me and scaffold_ok

        wants_cert = [x for x in (v.teacher_outcomes, v.student_outcomes, v.margin)
                      if x is not None]
        if wants_cert and len(wants_cert) < 3:
            raise ConfigError("the certificate needs all three of validate.teacher_outcomes, "
                              "validate.student_outcomes and validate.margin")
        if wants_cert:
            if v.margin > 0 or v.min_pairs < 1:
                raise ConfigError("validate.margin must be <= 0 and validate.min_pairs must be >= 1")
            cert, block = _stamp_certificate(Path(v.path), v)
            lines += ["", str(cert)]
            if v.per_task:
                lines += ["", "per-task (a macro average can hide a collapsed task):",
                          cert.per_task_table()]
            lines.append(f"\ncertificate stamped into {_declaration_path(Path(v.path))} "
                         f"(provenance.certificate — present, and never read by the runtime)")
            result["certificate"] = block
            ok = ok and cert.passed
        else:
            status, block = _verify_embedded_certificate(Path(v.path))
            if status == "intact":
                lines.append(f"  certificate: intact — {block['verdict']}, n={block['n_pairs']}, "
                             f"margin {block['margin_declared']:+.4f}, stamped {block['stamped_at']}")
                result["certificate"] = {"status": "intact", "verdict": block["verdict"]}
            elif status == "tampered":
                lines.append("  PROBLEM  embedded certificate fails its integrity hash — the "
                             "verdict has been edited after stamping")
                result["certificate"] = {"status": "tampered"}
                ok = False

        return CommandReport(result, "\n".join(lines), ok, 0 if ok else 1)

    return execute("validate", ValidateConfig, run, argv, prog="instinctflash validate",
                   description="Validate a checkpoint package; --validate.scaffold=<base|auto> "
                               "first writes its instinctflash.json from a built-in base "
                               "declaration; with outcome files, certify and stamp the "
                               "certificate into its provenance block.")


def cmd_plan(a) -> int:
    # DECLARATION-ONLY. `plan` answers "what would the runtime do to this checkpoint" and that
    # answer costs one small metadata file, never a weight snapshot and never a loaded model --
    # pinned by tests/test_cli_preflight.py. Building a Runtime here paid the full download.
    from instinctflash.runtime.facade import plan_declaration
    try:
        ckpt, _adapter, plan, probed = plan_declaration(
            a.model, strict=not a.any_checkpoint,
            tier_ceiling=a.tier_ceiling, exclude_passes=tuple(a.exclude_pass or ()))
    except Exception as e:                                       # noqa: BLE001
        print(f"{a.model}: {type(e).__name__}: {e}")
        return 1
    ex = ckpt.execution
    print(f"InstinctFlash plan preflight for {ex.model_id!r} (declaration-only; no weights fetched)")
    print(f"  declaration : {ckpt.path}")
    print(f"  backbone    : {ex.backbone}")
    print(f"  servable    : {ex.servable}")
    if probed is not None:
        cls, _why = probed.device_class()
        cap = (f"sm{probed.capability[0]}{probed.capability[1]}"
               if probed.capability != (0, 0) else "cpu")
        print(f"  device      : {probed.name}  {cap}  [{cls}]")
    else:
        print("  device      : not probed on this host")
    print(f"  capabilities: {', '.join(sorted(ckpt.capabilities()))}")
    print()
    print(plan.explain())
    return 0


def cmd_run(a) -> int:
    import time

    import numpy as np

    from instinctflash import Runtime
    try:
        rt = Runtime.from_pretrained(a.model)
    except Exception as e:                                       # noqa: BLE001
        print(f"{a.model}: {type(e).__name__}: {e}")
        return 1

    contract = rt.observation
    if contract is None or not contract.fields:
        print("This backbone declares no observation contract, so `run` cannot build a smoke-test "
              "input for it. Add ObservationSpec to its adapter, or use the Python API and pass a "
              "real observation.")
        return 2
    obs = contract.example()
    print(f"  expects: {contract.describe()}")

    # A prompt-conditioned model crashes deep in its forward when no prompt was ever encoded, so
    # the smoke test always supplies one. It conditions the actions, which a smoke test ignores.
    prompt = a.prompt or "smoke test: reach forward"
    print(f"SMOKE TEST -- zero-filled observations, prompt {prompt!r}. This proves the checkpoint "
          f"loads here and returns finite actions.\nIt is not an evaluation.\n")
    times, last = [], None
    with rt, rt.episode(prompt=prompt) as ep:
        for i in range(a.cycles):
            t0 = time.perf_counter()
            out = ep.predict(obs)
            times.append((time.perf_counter() - t0) * 1000)
            last = np.asarray(out["action"] if isinstance(out, dict) and "action" in out else out)
    warm = times[1:] or times
    print(f"  cycles      {len(times)}   first {times[0]:.1f} ms   warm median "
          f"{sorted(warm)[len(warm) // 2]:.1f} ms")
    print(f"  action      {last.shape} {last.dtype}")
    print(f"  finite      {bool(np.isfinite(last).all())}    std {float(last.std()):.4f}")
    return 0 if np.isfinite(last).all() else 1


def _serve_preflight(model: str, r: RuntimeConfig) -> tuple[dict, str]:
    """Device capabilities + declaration + plan, from one metadata file. Never weights."""
    from instinctflash.runtime.facade import plan_declaration

    ckpt, _adapter, plan, probed = plan_declaration(
        model, strict=True, nfe=r.nfe or None,
        tier_ceiling=r.tier_ceiling, exclude_passes=tuple(r.exclude_passes))
    ex = ckpt.execution
    if probed is not None:
        cap = f"sm{probed.capability[0]}{probed.capability[1]}" if probed.capability != (0, 0) else "cpu"
        mem = f"{probed.total_memory / 1e9:.0f} GB" if probed.total_memory else "-"
        device = f"{probed.name}  {cap}  {mem}"
        features = ", ".join(sorted(probed.features))
        dev_cls, dev_why = probed.device_class()
        device_class = f"{dev_cls} — {dev_why}"
    else:
        device = ("none visible — expected without the `runtime` extra; passes with hardware "
                  "requirements report APPLICABILITY UNCHECKED")
        features = ""
        device_class = ""
    lines = [
        f"InstinctFlash serve preflight for {ex.model_id!r} (declaration-only; no weights fetched)",
        f"  device      : {device}",
    ]
    if features:
        lines.append(f"  features    : {features}")
    if device_class:
        lines.append(f"  class       : {device_class}")
    if probed is not None and probed.device_class()[0] == "bandwidth-bound-edge":
        # Exactly one line, in the preflight rather than the README: the reader it is for is the
        # one who just saw their sm_110 plan decline capture with the measured reason.
        lines.append("  an engine tier for this device class is available under commercial "
                     "access — founders@general-instinct.com")
    lines += [
        f"  declaration : {ckpt.path}",
        f"  backbone    : {ex.backbone}",
        f"  servable    : {ex.servable}",
        f"  capabilities: {', '.join(sorted(ckpt.capabilities()))}",
        "",
        plan.explain(),
    ]
    result = {"model_id": ex.model_id, "backbone": ex.backbone, "servable": ex.servable,
              "device": device, "device_class": device_class,
              "capabilities": sorted(ckpt.capabilities()), "plan": plan.explain()}
    return result, "\n".join(lines)


def _serve_autoscaffold(model: str):
    """Scaffold a local checkpoint directory that has no declaration, inline, before preflight.

    Returns ``(scaffold_result | None, text, fill_me)``. A Hub id, a missing path, or a
    directory that already carries a declaration (or the legacy delta.json) returns
    ``(None, "", [])`` — serve behaves exactly as before. Everything else reuses the scaffold
    verb's own writer (`descriptors.scaffold.run_scaffold`, base=auto), so the announcement,
    the per-field evidence, and the written file are the same ones `validate` produces.

    Two walls, both BEFORE any download or load:

      * an unmerged peft/LoRA adapter is refused with `package.unmerged_adapter_problem`'s
        message — the exact merge command included — declaration or not, because serving that
        layout means serving the base without the fine-tune (the silent-wrong-model class).
      * FILL_ME sentinels stop the command — whether this run's scaffold just wrote them or an
        earlier one did (`fill_me_findings` runs on EVERY serve of a local directory, exactly as
        it does on every validate; without that, the second serve of a half-filled scaffold would
        download the base's frozen stack and only then hit the adapter's geometry refusal). The
        caller stops with each missing field, the scaffold's one-line "where the value comes
        from", and the rerun. Never guess, never serve a wrong-geometry model.

    A directory whose declaration is complete returns ``(None, "", [])`` and serve proceeds
    exactly as it always has.
    """
    from instinctflash.cli_config import ConfigError

    d = Path(model)
    if not d.is_dir():
        return None, "", []
    from instinctflash.descriptors.package import unmerged_adapter_problem
    problem = unmerged_adapter_problem(d)
    if problem is not None:
        raise ConfigError(f"{d}: {problem}\n    then serve the MERGED output.")

    from instinctflash.descriptors.checkpoint import _declaration_file
    from instinctflash.descriptors.scaffold import ScaffoldError, fill_me_findings, run_scaffold
    sres = None
    lines: list[str] = []
    if _declaration_file(d) is None and not (d / "delta.json").is_file():
        try:
            sres, stext, _wrote = run_scaffold(d, "auto")
        except ScaffoldError as e:
            raise ConfigError(str(e)) from e
        lines = [f"no declaration in {d} — scaffolding one from the checkpoint's own evidence "
                 f"(the same writer as `instinctflash validate --validate.scaffold=auto`):",
                 stext]
    fill_me = fill_me_findings(d)
    if fill_me:
        wrote_now = "The declaration was written" if sres is not None \
            else f"Its declaration ({_declaration_file(d).name}) still carries"
        lines += [
            *([""] if lines else []),
            f"SERVE STOPPED — before any download or load: {len(fill_me)} field(s) this "
            f"checkpoint cannot prove, and the runtime never guesses. {wrote_now} "
            f"FILL_ME at each:",
            *(f"  {where} — {why}" for where, why in fill_me),
            "",
            f"Fill each value in {_declaration_file(d)}, then rerun:",
            f"  instinctflash serve {d}",
        ]
    return sres, "\n".join(lines), fill_me


def _serve_smoke(rt, preflight: dict):
    """One zero-filled control cycle: does this checkpoint load HERE and return finite actions."""
    import numpy as np

    from instinctflash.cli_config import CommandReport

    contract = rt.observation
    if contract is None or not contract.fields:
        return CommandReport(
            preflight,
            "This backbone declares no observation contract, so the smoke test cannot build an "
            "input for it. Add ObservationSpec to its adapter, or use the Python API and pass a "
            "real observation.", False, 2)
    obs = contract.example()
    geometry_source = rt.observation_source
    with rt.episode(prompt="smoke test: reach forward") as ep:
        out = ep.predict(obs)
    last = np.asarray(out["action"] if isinstance(out, dict) and "action" in out
                      else out.get("actions") if isinstance(out, dict) else out)
    finite = bool(np.isfinite(last).all())
    text = ("SMOKE TEST — one zero-filled observation, prompt 'smoke test: reach forward'. This "
            "proves the checkpoint loads here and returns finite actions. It is not an evaluation.\n"
            f"  expects  {contract.describe()}\n"
            f"  geometry {geometry_source}\n"
            f"  action   {last.shape} {last.dtype}   finite {finite}   std {float(last.std()):.4f}")
    return CommandReport(
        {**preflight, "smoke": {"action_shape": list(last.shape), "dtype": str(last.dtype),
                                "finite": finite,
                                "observation_source": geometry_source}},
        text, finite, 0 if finite else 1)


def cmd_serve(argv: list[str]) -> int:
    """`instinctflash serve <model-id>` — preflight, then the openpi-wire websocket policy server.

    A local directory with no declaration scaffolds its own first (`_serve_autoscaffold`): the
    detected base, every inherited/inferred field with its evidence, and the written file are
    all announced, so the one command from "training output" to "server up" stays inspectable.
    Preflight prints BEFORE any weight moves. Then load-then-bind, in that order: a checkpoint
    that cannot load exits here with the loader's error and the port never opens, because the
    openpi client retries a dead port forever and a half-started server is the documented worst
    case (eval/lingbot_va_robotwin/README.md).
    """
    from instinctflash.cli_config import (
        CommandReport, ConfigError, UnsupportedCapability, execute,
    )

    # `serve <model-id> --serve.port=...`: the leading positional is sugar for --serve.model=.
    if argv and not argv[0].startswith("-"):
        argv = [f"--serve.model={argv[0]}", *argv[1:]]

    def run(cfg: ServeConfig) -> CommandReport:
        s = cfg.serve
        if not s.model:
            raise ConfigError("serve.model is required: instinctflash serve <model-id>")
        scaffold, scaffold_text, fill_me = _serve_autoscaffold(s.model)
        if fill_me:
            # STOP, loudly and completely, before any download: the text above already carries
            # each missing field, why the scaffold refused to guess it, and the exact rerun.
            return CommandReport(
                {"model": s.model, "scaffold": scaffold,
                 "fill_me": [where for where, _ in fill_me]},
                scaffold_text, False, 1)
        try:
            preflight, preflight_text = _serve_preflight(s.model, cfg.runtime)
        except Exception:
            # The scaffold just WROTE a declaration into the user's directory; a preflight
            # failure (unknown backbone, refused plan) must not swallow that announcement —
            # the user needs to know the file exists, what was inferred, and on what evidence.
            if scaffold_text:
                print(scaffold_text + "\n", file=sys.stderr)
            raise
        if scaffold is not None:
            preflight["scaffold"] = scaffold
            preflight_text = scaffold_text + "\n\n" + preflight_text
        if s.dry_run:
            return CommandReport(preflight, preflight_text, True, 0)
        # From here on the command runs for a while (or forever): the preflight and the logs go
        # to stderr NOW rather than through execute()'s deferred stdout capture.
        print(preflight_text + "\n", file=sys.stderr)

        viz = None
        if s.viz and not s.smoke:
            from instinctflash.serving.viz import RerunViz
            try:
                # BEFORE the model loads: discovering a missing extra after a 10 GB download is
                # the wrong order.
                viz = RerunViz(session_name=f"instinctflash-serve {s.model}", sink=s.viz_sink)
            except ImportError as e:
                raise UnsupportedCapability(
                    "--serve.viz needs rerun-sdk: pip install 'instinctflash[viz]'") from e

        import logging
        logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                            format="%(asctime)s %(name)s %(message)s")
        from instinctflash import Runtime

        r = cfg.runtime
        rt = Runtime.from_pretrained(
            s.model, device=r.device, placement=r.placement, nfe=r.nfe or None,
            tier_ceiling=r.tier_ceiling, exclude_passes=tuple(r.exclude_passes),
            seed=s.seed)
        try:
            if s.smoke:
                return _serve_smoke(rt, preflight)
            from instinctflash.serving import WebsocketPolicyServer
            server = WebsocketPolicyServer(rt, host=s.host, port=s.port, viz=viz)
            print(f"loaded {rt.model_id!r}; binding ws://{s.host}:{s.port} "
                  f"(clients: openpi_client.WebsocketClientPolicy, or GET /healthz)",
                  file=sys.stderr)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        finally:
            if viz is not None:
                viz.close()
            rt.close()
        return CommandReport({**preflight, "host": s.host, "port": s.port},
                             "server stopped", True, 0)

    return execute("serve", ServeConfig, run, argv, prog="instinctflash serve",
                   description="Preflight a checkpoint, then serve it over websocket on the "
                               "openpi wire protocol (--serve.dry_run / --serve.smoke stop "
                               "early). A local directory with no declaration scaffolds its "
                               "own first and stops before any download if facts are missing.")


def cmd_certify(argv: list[str]) -> int:
    """Paired non-inferiority certificate from two outcome JSONL files.

    A thin CLI over `instinctflash.verify.certify` — the same code path the harnesses use, so a
    certificate produced here is the certificate, not a reimplementation of one.
    """
    from dataclasses import asdict

    from instinctflash.cli_config import CommandReport, ConfigError, execute

    def run(cfg: CertifyConfig) -> CommandReport:
        c = cfg.certify
        if c.teacher_outcomes is None or c.student_outcomes is None or c.margin is None:
            raise ConfigError(
                "certify.teacher_outcomes, certify.student_outcomes, and certify.margin are "
                "required")
        if c.margin > 0 or c.min_pairs < 1:
            raise ConfigError("certify.margin must be <= 0 and certify.min_pairs must be >= 1")
        from instinctflash.verify.certify import certify, load_jsonl

        cert = certify(
            load_jsonl(str(c.teacher_outcomes)), load_jsonl(str(c.student_outcomes)),
            margin=c.margin, min_pairs=c.min_pairs,
            teacher_hash=c.teacher_hash, student_hash=c.student_hash,
            harness=c.harness or "?", recipe=c.recipe or "?",
            seeds=",".join(map(str, c.seeds)) if c.seeds is not None else "?",
        )
        result = asdict(cert)
        result["passed"] = bool(cert.passed)
        if not c.per_task:
            result.pop("per_task", None)
        text = str(cert)
        if c.per_task:
            text += ("\n\nper-task (a macro average can hide a collapsed task):\n"
                     + cert.per_task_table())
        return CommandReport(result, text, cert.passed, 0 if cert.passed else 1)

    return execute("certify", CertifyConfig, run, argv, prog="instinctflash certify",
                   description="Certify paired teacher/student outcomes at a declared margin.")


#: one-line pointers the compatibility aliases print (to stderr) before delegating.
_ALIAS_POINTERS = {
    "devices": "its report is part of `instinctflash serve <model-id> --serve.dry_run=true`",
    "describe": "use `instinctflash serve <model-id> --serve.dry_run=true`",
    "plan": "use `instinctflash serve <model-id> --serve.dry_run=true`",
    "run": "use `instinctflash serve <model-id> --serve.smoke=true`",
    "certify": "use `instinctflash validate <dir> --validate.teacher_outcomes=... "
               "--validate.student_outcomes=... --validate.margin=...` (also stamps the "
               "certificate into the package)",
}


def _alias_note(verb: str) -> None:
    print(f"note: `instinctflash {verb}` is a compatibility alias — {_ALIAS_POINTERS[verb]}",
          file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The two verbs speak the typed dotted-field syntax and own their help/error contract, so
    # they are dispatched before argparse (whose positional grammar the aliases keep).
    if argv[:1] == ["serve"]:
        return cmd_serve(argv[1:])
    if argv[:1] == ["validate"]:
        return cmd_validate(argv[1:])
    if argv[:1] == ["certify"]:
        _alias_note("certify")
        return cmd_certify(argv[1:])

    ap = argparse.ArgumentParser(prog="instinctflash", description=__doc__.split("\n")[0])
    # metavar hides the alias verbs from usage; registering them without help= hides them from
    # the listing. They still parse — existing scripts keep working — they are just not taught.
    sub = ap.add_subparsers(dest="cmd", metavar="{serve,validate}")

    sub.add_parser("serve", help="deploy: preflight (device + declaration + plan), then serve "
                                 "over websocket on the openpi wire protocol; a local fine-tune "
                                 "directory with no declaration scaffolds its own first; "
                                 "--serve.dry_run and --serve.smoke stop early "
                                 "(see `instinctflash serve -h`)")

    sub.add_parser("validate", help="trust: is this directory a publishable checkpoint; "
                                    "--validate.scaffold=<base|auto> writes its instinctflash.json "
                                    "from a built-in base first; with "
                                    "--validate.teacher_outcomes/.student_outcomes/.margin also "
                                    "certifies and stamps the certificate into the package "
                                    "(see `instinctflash validate -h`)")

    # -- undocumented compatibility aliases below this line ----------------------------------------
    sub.add_parser("devices").set_defaults(fn=cmd_devices)

    d = sub.add_parser("describe")
    d.add_argument("model")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_describe)

    p = sub.add_parser("plan")
    p.add_argument("model")
    p.add_argument("--any-checkpoint", action="store_true",
                   help="do not refuse a checkpoint declaring servable=false")
    p.add_argument("--tier-ceiling", choices=("bitexact", "numeric", "behavioral"),
                   default="bitexact",
                   help="the strongest accuracy claim the plan may spend (a claim budget, "
                        "not a speed knob)")
    p.add_argument("--exclude-pass", action="append", metavar="NAME",
                   help="drop a pass via Plan.without(); a caller exclusion the runtime honors "
                        "everywhere (repeatable)")
    p.set_defaults(fn=cmd_plan)

    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--cycles", type=int, default=3)
    r.add_argument("--prompt", default=None)
    r.set_defaults(fn=cmd_run)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 2
    if a.cmd in _ALIAS_POINTERS:
        _alias_note(a.cmd)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
