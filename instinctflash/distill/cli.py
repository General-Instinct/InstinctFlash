"""`python -m instinctflash.distill` — training and verification commands.

    python -m instinctflash.distill steps <ckpt> <out> --schedule video=1,action=4 \
        --dataset your_org/your_demos

The module command connects screening, training and verification, with standard-format
checkpoint output. It is separate from the inference CLI.

What runs today (the skeleton): family resolution from the checkpoint's own declaration,
stream/grid/trainable-set planning on CPU, the capability gate, screening-spec emission
(--emit-screen-spec, with the guidance axis), the screen VERDICT from a frontier report
(--screen-report: NO DISTILLATION NEEDED writes the frontier + intervals + a certification
prereg stub and stops; TRAIN proceeds), and the three-arm verify with the control gate
(--control-grid: the guidance grid the control was the best of). The train stage stops at each
family's ExperimentalSeam until its campaign lands a recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_schedule(text: str) -> dict[str, int]:
    """'video=1,action=4' -> {'video': 1, 'action': 4}. Per-stream, never a bare count."""
    out: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(
                f"--schedule entries are stream=steps (got {item!r}): a bare count would "
                "silently apply one step count to every stream, which is the failure mode the "
                "per-stream declaration exists to prevent"
            )
        name, _, steps = item.partition("=")
        out[name.strip()] = int(steps)
    if not out:
        raise SystemExit("--schedule must name at least one stream, e.g. video=1,action=4")
    return out


def _parse_guidance(text: str | None) -> dict | None:
    """'video=1,action=positive_only' -> {'video': 1.0, 'action': 'positive_only'} (declaration schema)."""
    if not text:
        return None
    out: dict = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--guidance entries are stream=value (got {item!r})")
        name, _, value = item.partition("=")
        value = value.strip()
        try:
            out[name.strip()] = float(value)
        except ValueError:
            out[name.strip()] = value
    return out or None


def _parse_guidance_grid(text: str | None) -> dict | None:
    """'video=1/3/5/7/9' -> {'video': [1.0, 3.0, 5.0, 7.0, 9.0]}."""
    if not text:
        return None
    out: dict = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, values = item.partition("=")
        out[name.strip()] = [float(v) for v in values.split("/") if v.strip()]
    return out or None


def _family_for(ckpt: str, override: str | None) -> str:
    if override:
        return override
    path = Path(ckpt)
    if path.is_dir():
        from instinctflash.descriptors.checkpoint import load_declaration

        backbone = load_declaration(path).backbone
        if backbone:
            return backbone
    from instinctflash.distill.adapter import registered_families

    raise SystemExit(
        f"{ckpt}: cannot resolve a model family (no local declaration); pass --family "
        f"explicitly (registered families: {', '.join(sorted(registered_families()))})"
    )


def _cmd_steps(args) -> int:
    from instinctflash.distill.adapter import ExperimentalSeam, get_family
    from instinctflash.distill.pipeline import DistillPipeline
    from instinctflash.train.recipe import Environment

    nfe = _parse_schedule(args.schedule)
    family = _family_for(args.checkpoint, args.family)
    adapter = get_family(family)
    guidance = _parse_guidance(args.guidance)
    pipeline = DistillPipeline(
        adapter, nfe, guidance=guidance, out_dir=args.out, dataset=args.dataset,
        recipe_id=args.recipe_id,
    )

    point = pipeline.student_schedule()
    print(f"few-step plan for {args.checkpoint!r} (family {family}, target operating point: "
          f"schedule " + ", ".join(f"{k}={v}" for k, v in sorted(nfe.items()))
          + " | guidance " + ", ".join(f"{s}={g['mode']}@{g['scale']:g}" if g.get('scale') is not None
                                       else f"{s}={g['mode']}" for s, g in sorted(point["guidance"].items()))
          + ")")
    for spec in adapter.streams():
        target = nfe.get(spec.name)
        trainable = adapter.trainable_set(spec.name)
        print(f"  stream {spec.name}: teacher {spec.teacher_nfe} steps, shift {spec.shift:g}, "
              f"guidance {spec.guidance_mode}({spec.guidance_scale:g})"
              + (f", certified untrained at {spec.certified_nfe}" if spec.certified_nfe else ""))
        if target is not None:
            print(f"    grid    : {pipeline.grids[spec.name].describe()}")
        print("    training: " + ("UNTOUCHED by construction — " + trainable.note
                                  if trainable.untouched
                                  else f"unfreeze {list(trainable.unfreeze)} — {trainable.note}"))
    ok, why = adapter.requires().satisfied_by(Environment())
    print(f"  capabilities: {'ok' if ok else 'REFUSED'} — {why}")
    if not ok:
        return 1

    if args.emit_screen_spec:
        spec = pipeline.screening_spec(
            name=f"{family} fewstep screen ({args.schedule})",
            driver={
                "command": ["${IFL_SWEEP_DRIVER_PYTHON}", "${IFL_SWEEP_DRIVER}"],
                "environment": {},
                "timeout_seconds": 7200,
                "revision": "${IFL_SWEEP_DRIVER_REVISION}",
            },
            guidance_grid=_parse_guidance_grid(args.guidance_grid),
        )
        Path(args.emit_screen_spec).write_text(json.dumps(spec, indent=2) + "\n")
        print(f"\nscreening spec written to {args.emit_screen_spec} "
              f"({len(spec['points'])} operating points"
              + (f"; guidance axis {spec['guidance_axis']}" if spec.get("guidance_axis") else "")
              + "); run it with:")
        print(f"  python -m benchmarks.vla sweep-plan --sweep {args.emit_screen_spec} "
              "--profile <profile> --output plan.json")
        print("  python -m benchmarks.vla run --plan plan.json --output run/ --gpu <n>")
        print("  python -m benchmarks.vla sweep-report --run run/")
        print("its rows are the standing matched-NFE controls; per schedule the control is the best "
              "row over the guidance grid. Then: steps ... --screen-report run/frontier.json")
        return 0

    from instinctflash.distill.pipeline import ControlGateViolation, NoDistillationNeeded

    screen = None
    if args.screen_report:
        report = json.loads(Path(args.screen_report).read_text())
        screen = pipeline.screen_verdict(report, source=str(args.screen_report))
        print(f"\nscreen verdict: {screen.headline()}")
        # the verdict's artifact is written whatever it says: frontier rows + intervals at the
        # target schedule, and the certification prereg stub for the best untrained point
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        artifact = screen.artifact()
        name = "no_distillation_needed" if screen.no_distillation_needed else "screen_verdict"
        (out / f"{name}.json").write_text(json.dumps(artifact, indent=2) + "\n")
        (out / f"{name}.md").write_text(
            f"# {screen.headline()}\n\n" + (artifact["frontier_table_markdown"] or "")
            + "\n\n" + artifact["certification_prereg_stub_markdown"] + "\n")
        print(f"screen artifact written to {out}/{name}.{{json,md}}")
    try:
        pipeline.train(teacher=None, screen=screen)  # the skeleton's teacher loading lands with the recipe
    except NoDistillationNeeded:
        print("\nNO DISTILLATION NEEDED — no trainer run; the artifact above is what this verdict owes "
              "(frontier + intervals + certification prereg stub for the untrained point).")
        return 0
    except ControlGateViolation as refusal:
        print(f"\nREFUSED: {refusal}", file=sys.stderr)
        return 4
    except ExperimentalSeam as seam:
        print(f"\nTRAIN STAGE SEAMED — {seam}")
        return 3
    return 0


def _cmd_verify(args) -> int:
    from instinctflash.distill.pipeline import ControlGateViolation, DistillPipeline

    nfe = _parse_schedule(args.schedule)
    pipeline = DistillPipeline(
        args.family, nfe, guidance=_parse_guidance(args.guidance), out_dir=args.out,
        dataset=args.dataset, recipe_id=args.recipe_id,
    )
    control_schedule = None
    if args.control_schedule:
        control_schedule = json.loads(Path(args.control_schedule).read_text())
    control_grid = None
    if args.control_grid:
        control_grid = json.loads(Path(args.control_grid).read_text())
        if isinstance(control_grid, dict) and "candidates" in control_grid:
            control_grid = control_grid["candidates"]  # a best_untrained_per_schedule entry
    try:
        report = pipeline.verify_and_stamp(
            point=args.point or args.schedule,
            teacher_outcomes=args.teacher_outcomes,
            control_outcomes=args.control_outcomes,
            student_outcomes=args.student_outcomes,
            control_schedule=control_schedule,
            control_grid=control_grid,
            margin=args.margin,
            interval=args.interval,
            min_pairs=args.min_pairs,
        )
    except ControlGateViolation as violation:
        print(f"REFUSED: {violation}", file=sys.stderr)
        return 2
    print(report.summary())
    if args.out:
        print(f"\nprovenance.distillation stamped into {args.out} "
              "(verified by every later `instinctflash validate`)")
    passed = report.c_minus_b.passed and report.c_minus_a.passed
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m instinctflash.distill",
        description=__doc__.split("\n")[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    steps = sub.add_parser(
        "steps", help="distill a checkpoint to a declared few-step schedule (screen->train->verify)"
    )
    steps.add_argument("checkpoint")
    steps.add_argument("out")
    steps.add_argument("--schedule", required=True, help="per-stream, e.g. video=1,action=4")
    steps.add_argument(
        "--guidance", default=None,
        help="the student's target guidance leg, per stream, in the declaration schema "
             "(e.g. video=1,action=positive_only); default = the family's own",
    )
    steps.add_argument(
        "--guidance-grid", default=None,
        help="scales to screen per stream, e.g. video=1/3/5/7/9 (default on a CFG family: the "
             "family scale and 1.0)",
    )
    steps.add_argument(
        "--screen-report", metavar="PATH", default=None,
        help="a sweep-report frontier.json; decides NO DISTILLATION NEEDED / TRAIN / INCONCLUSIVE "
             "at the target schedule before any trainer runs",
    )
    steps.add_argument("--dataset", default=None)
    steps.add_argument("--family", default=None, help="override the declared backbone")
    steps.add_argument("--recipe-id", default="fewstep_v0")
    steps.add_argument(
        "--emit-screen-spec", metavar="PATH", default=None,
        help="write the schedule-sweep spec (the matched-NFE controls) and stop",
    )
    steps.set_defaults(fn=_cmd_steps)

    verify = sub.add_parser(
        "verify",
        help="three-arm verification (A teacher / B untrained matched-NFE control / C student) "
        "+ provenance stamp; refuses without the control",
    )
    verify.add_argument("--family", required=True)
    verify.add_argument("--schedule", required=True)
    verify.add_argument("--guidance", default=None, help="the student's guidance leg (declaration schema)")
    verify.add_argument(
        "--control-grid", default=None,
        help="JSON: the guidance grid swept at the control's schedule (a sweep-report "
             "best_untrained_per_schedule entry, or its candidates list); the control must be its best",
    )
    verify.add_argument("--point", default=None)
    verify.add_argument("--teacher-outcomes", required=True)
    verify.add_argument("--control-outcomes", default=None)
    verify.add_argument("--student-outcomes", required=True)
    verify.add_argument(
        "--control-schedule", default=None,
        help="JSON file declaring the control arm's execution configuration",
    )
    verify.add_argument("--margin", type=float, required=True)
    verify.add_argument("--interval", default="wald_central95")
    verify.add_argument("--min-pairs", type=int, default=1)
    verify.add_argument("--out", default=None, help="package dir to stamp provenance into")
    verify.add_argument("--dataset", default=None)
    verify.add_argument("--recipe-id", default="fewstep_v0")
    verify.set_defaults(fn=_cmd_verify)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
