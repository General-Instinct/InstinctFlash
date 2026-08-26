"""`instinctflash` — the command line. Six verbs, no Python required.

    instinctflash devices                    what machine am I on, and what can it do
    instinctflash describe  <model-id>       what a checkpoint declares, without downloading weights
    instinctflash validate  <dir>            is this directory a publishable checkpoint
    instinctflash plan      <model-id>       what the runtime would do to it, and why
    instinctflash run       <model-id>       load it and produce real actions
    instinctflash certify   --certify....    paired non-inferiority certificate from two outcome files

Why a CLI at all, given `Runtime.from_pretrained` is three lines. Because "install the package, give
it a model id, run" should not require writing a program, and because most of these verbs answer
questions you want answered BEFORE committing to a download or a GPU: what is this checkpoint, will
this runtime serve it, what would it do to it, and is this machine capable of the plan. `plan` and
`describe` need no weights and no GPU at all.

`certify` uses the typed dotted-field syntax from `cli_config` (`--certify.margin=-0.05`,
optional `--config_path=FILE` with CLI overrides winning, unknown fields are hard errors, JSON
errors use one stable schema, `--output.path` writes atomically). The classic verbs keep their
existing syntax — that surface is published and stays stable.

`run` uses zero-filled observations by default. That is deliberately a smoke test and says so: it
proves this checkpoint loads on this machine and produces finite actions of the right shape, which is
the question a new user actually has. It is not an evaluation and the output is not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from instinctflash.cli_config import OutputConfig


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
    return f"{d.name}  {cap}  {mem}\n  features: {', '.join(sorted(d.features))}"


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


def cmd_validate(a) -> int:
    from instinctflash.descriptors.package import (
        publishability, validate_package, verify_weights_indexes,
    )
    rep = validate_package(a.path)
    print(rep.explain())
    # Weights-index integrity GATES the exit code: a package whose declared shards are missing,
    # or whose shard paths escape the package, is not a valid package — the old exit simply could
    # not see it because validation read declarations, not weights. Publishability stays
    # INFORMATIONAL (exit-neutral), preserving the published exit contract for packages that are
    # valid but carry training internals.
    index_problems = verify_weights_indexes(a.path)
    for p in index_problems:
        print(f"  PROBLEM  {p}")
    ok = False
    try:
        ok, findings = publishability(a.path)
        print(f"  publishable without training internals: {'YES' if ok else 'NO'}")
        for f in findings:
            print(f"    - {f}")
    except Exception as e:                                       # noqa: BLE001
        print(f"  publishability: {type(e).__name__}: {e}")
    return 0 if rep.ok and not index_problems else 1


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
    print(f"  device      : {probed.name if probed is not None else 'not probed on this host'}")
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # certify speaks the typed dotted-field syntax and owns its own help/error contract, so it is
    # dispatched before argparse (whose positional grammar the classic verbs keep).
    if argv[:1] == ["certify"]:
        return cmd_certify(argv[1:])

    ap = argparse.ArgumentParser(prog="instinctflash", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("certify", help="paired non-inferiority certificate (typed "
                                   "--certify.field=value syntax; see `instinctflash certify -h`)")

    sub.add_parser("devices", help="what machine am I on").set_defaults(fn=cmd_devices)

    d = sub.add_parser("describe", help="what a checkpoint declares, without its weights")
    d.add_argument("model")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_describe)

    v = sub.add_parser("validate", help="is this directory a publishable checkpoint")
    v.add_argument("path")
    v.set_defaults(fn=cmd_validate)

    p = sub.add_parser("plan", help="what the runtime would do, and why (declaration-only)")
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

    r = sub.add_parser("run", help="load it and produce real actions (smoke test)")
    r.add_argument("model")
    r.add_argument("--cycles", type=int, default=3)
    r.add_argument("--prompt", default=None)
    r.set_defaults(fn=cmd_run)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
