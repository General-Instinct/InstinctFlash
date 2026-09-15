"""Command-line entrypoint for the reproducible VLA benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .doctor import inspect_plan
from .plan import build_plan, load_arms, validate_plan
from .registry import load_registry
from .report import build_report
from .runner import execute_plan
from .schedule_sweep import build_sweep_plan, build_sweep_report, load_sweep
from .util import ConfigurationError, load_json, write_json_atomic


def _models(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    selected = []
    for value in values:
        selected.extend(item for item in value.split(",") if item)
    return selected


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.vla",
        description="Immutable paired benchmarks for VLA acceleration and quantization",
    )
    parser.add_argument("--registry", type=Path, default=None, help="alternate registry JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    startup = subparsers.add_parser("startup-probe", help="fresh-process native admission receipt, including rejected capture")
    startup.add_argument("--model",required=True)
    startup.add_argument("--revision",required=True)
    startup.add_argument("--mode",choices=["stock","runtime_default"],required=True)
    startup.add_argument("--output",type=Path,required=True)
    startup.add_argument("--checkpoint",type=Path)
    startup.add_argument("--subdir",default="libero_10")
    startup.add_argument("--require-capture",action="store_true")
    startup.add_argument("--tier-ceiling",choices=["bitexact","numeric"])

    qualify = subparsers.add_parser("qualification-report", help="checkpoint-specific startup, simulator and target-device evidence inventory")
    qualify.add_argument("--target-device",required=True)
    qualify.add_argument("--bundle",type=Path,action="append",default=[])
    qualify.add_argument("--record",type=Path,action="append",default=[])
    qualify.add_argument("--startup-receipt",type=Path,action="append",default=[])
    qualify.add_argument("--minimum-startups",type=int,default=3)
    qualify.add_argument("--output",type=Path,required=True)

    record = subparsers.add_parser("execution-report", help="bind exact execution identity to verified timing and quality evidence")
    record.add_argument("--receipt",type=Path,required=True)
    record.add_argument("--measurement",type=Path)
    record.add_argument("--bundle",type=Path,action="append",default=[])
    record.add_argument("--output",type=Path,required=True)
    choose = subparsers.add_parser("select-configuration", help="select only within explicit budgets, precision permissions and matching quality evidence")
    choose.add_argument("--record",type=Path,action="append",required=True)
    choose.add_argument("--budget",type=Path,required=True)
    choose.add_argument("--policy",type=Path,required=True)
    choose.add_argument("--output",type=Path,required=True)
    endpoint = subparsers.add_parser("measure-endpoint", help="measure a pinned live policy using real recorded observations")
    endpoint.add_argument("--trace",type=Path,required=True)
    endpoint.add_argument("--endpoint",required=True)
    endpoint.add_argument("--receipt",type=Path,required=True)
    endpoint.add_argument("--warmup",type=int,default=8)
    endpoint.add_argument("--iterations",type=int,default=128)
    endpoint.add_argument("--output",type=Path,required=True)
    budget = subparsers.add_parser("realtime-report", help="assess actual device samples against an explicit control budget")
    budget.add_argument("--measurement", type=Path, required=True)
    budget.add_argument("--budget", type=Path, required=True)
    budget.add_argument("--output", type=Path, required=True)
    latency = subparsers.add_parser("measure-policy", help="measure native policy calls on this device with recorded real inputs")
    latency.add_argument("--trace",type=Path,required=True)
    latency.add_argument("--model",required=True)
    latency.add_argument("--revision",required=True)
    latency.add_argument("--mode",choices=['stock','runtime_default'],required=True)
    latency.add_argument("--warmup",type=int,default=8)
    latency.add_argument("--iterations",type=int,default=128)
    latency.add_argument("--output",type=Path,required=True)
    latency.add_argument("--tier-ceiling", choices=["bitexact", "numeric"])
    view = subparsers.add_parser("checkpoint-view", help="declare a verified view of a supported native checkpoint")
    view.add_argument("--model", required=True)
    view.add_argument("--revision", required=True)
    view.add_argument("--subdir", required=True)
    view.add_argument("--output", type=Path, required=True)
    fixed = subparsers.add_parser("replay-policy", help="run real recorded observations with explicit recorded noise")
    fixed.add_argument("--trace", type=Path, action="append", required=True)
    fixed.add_argument("--endpoint", required=True)
    fixed.add_argument("--receipt", type=Path, required=True)
    fixed.add_argument("--repeats", type=int, default=3)
    fixed.add_argument("--output", type=Path, required=True)
    export = subparsers.add_parser("export", help="export a completed run with its immutable evidence")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-bundle", help="verify portable evidence and raw result identities")
    verify.add_argument("--bundle", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-scenes", help="prepare frozen scenes in isolated resumable task workers")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repo-root", type=Path, default=Path.cwd())
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--gpu", action="append", default=[])
    prepare.add_argument("--reuse-scenes", type=Path, action="append", default=[])

    campaign = subparsers.add_parser("sim-plan", help="build a frozen simulator screening campaign from an explicit spec")
    campaign.add_argument("--spec", type=Path, required=True)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--scenes", type=Path, default=None)

    sim_doctor = subparsers.add_parser("simulator-doctor", help="inspect local simulator rendering hardware")
    sim_doctor.add_argument("--simulator", choices=["libero", "robotwin", "robolab", "droid_sim_evals"], required=True)

    cov = subparsers.add_parser("coverage", help="show simulator routes, implemented adapters and validated run evidence")
    cov.add_argument("--run", type=Path, action="append", default=[], help="completed run directory; repeat for multiple campaigns")

    subparsers.add_parser("adapters", help="list explicit model/simulator contracts as JSON")

    registry = subparsers.add_parser("registry", help="validate and print the supported matrix")
    registry.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan", help="compile an immutable, counterbalanced paired plan")
    # profiles are registry facts (including --registry variants); build_plan rejects unknown ones
    plan.add_argument("--profile", required=True)
    plan.add_argument("--arms", type=Path, required=True)
    plan.add_argument("--model", action="append", help="model id; repeat or comma-separate")
    plan.add_argument("--output", type=Path, required=True)

    doctor = subparsers.add_parser("doctor", help="preflight drivers, locks, caches, and coverage")
    doctor.add_argument("--plan", type=Path, required=True)
    doctor.add_argument("--require-cached", action="store_true")

    run = subparsers.add_parser("run", help="execute or resume a plan without a shell")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--gpu", default=None, help="physical/visible GPU selector for CUDA_VISIBLE_DEVICES")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--paired-workers", type=int, choices=(1, 2), default=1, help="run two closed-loop arms concurrently on independent remote endpoints")

    report = subparsers.add_parser("report", help="validate evidence and apply release gates")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path, default=None)
    report.add_argument("--allow-synthetic", action="store_true", help="CI only; never use for claims")

    sweep_plan = subparsers.add_parser(
        "sweep-plan",
        help="compile a schedule-sweep spec (baseline + N schedule points, paired, "
        "counterbalanced) into an immutable plan; run it with the ordinary `run` verb",
    )
    sweep_plan.add_argument("--sweep", type=Path, required=True, help="sweep spec JSON")
    sweep_plan.add_argument("--profile", required=True)
    sweep_plan.add_argument("--model", action="append", help="model id; repeat or comma-separate")
    sweep_plan.add_argument("--output", type=Path, required=True)

    sweep_report = subparsers.add_parser(
        "sweep-report",
        help="pair every schedule point against the baseline arm and emit the frontier table "
        "(SCREENING: intervals only, no ship verdicts)",
    )
    sweep_report.add_argument("--run", type=Path, required=True)
    sweep_report.add_argument("--output", type=Path, default=None)
    sweep_report.add_argument(
        "--allow-synthetic", action="store_true", help="CI only; never use for claims"
    )

    prefetch = subparsers.add_parser("prefetch", help="materialize locked Hub revisions")
    prefetch.add_argument("--model", action="append", help="model id; defaults to all built-ins")
    prefetch.add_argument("--datasets", action="store_true")
    prefetch.add_argument("--execute", action="store_true", help="download; default is a dry run")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        if args.command == "startup-probe":
            if args.model in ('robbyant/lingbot-va-posttrain-robotwin', 'robbyant/lingbot-va-posttrain-libero-long'):
                if args.require_capture or args.checkpoint is not None or args.subdir != 'libero_10' or args.tier_ceiling not in (None, 'bitexact'):
                    raise ConfigurationError('VA startup preserves native precision/schedule and uses its deferred-commit protocol, not capture admission')
                if args.mode == 'stock' and args.tier_ceiling is not None:
                    raise ConfigurationError('stock does not accept a Runtime tier policy')
                from .wan_va_startup import probe
                probe(args.model, args.revision, args.mode, args.output)
                _print({'output': str(args.output), 'scope': 'native prediction/commit startup only'})
                return 0
            from .joint_policy_server import MODELS,main as joint_startup
            if args.model in MODELS:
                if args.checkpoint is not None or args.subdir != 'libero_10':
                    raise ConfigurationError('joint startup uses its pinned native checkpoint, not a GR00T view')
                command=['--model',args.model,'--revision',args.revision,'--mode',args.mode,
                         '--receipt',str(args.output),'--port','0','--startup-only']
                if args.require_capture:command.append('--require-capture')
                if args.tier_ceiling is not None:command += ['--tier-ceiling',args.tier_ceiling]
                joint_startup(command)
            else:
                from .startup_probe import probe
                if args.tier_ceiling not in (None,'bitexact') or (args.mode=='stock' and args.tier_ceiling is not None):
                    raise ConfigurationError('this startup contract preserves its native BITEXACT transformation ceiling')
                probe(args.model,args.revision,args.mode,args.output,checkpoint=args.checkpoint,
                      subdir=args.subdir,require_capture=args.require_capture)
            _print({'output':str(args.output),'scope':'startup admission only'})
            return 0
        if args.command == "qualification-report":
            from .qualification import report
            if args.output.exists():raise ConfigurationError('refusing to overwrite qualification report')
            value=report(registry,target_device=args.target_device,bundles=args.bundle,
                records=[load_json(p) for p in args.record],startup_receipts=args.startup_receipt,
                minimum_startups=args.minimum_startups)
            write_json_atomic(args.output,value)
            _print({'output':str(args.output),'checkpoints':len(value['checkpoints']),'deployment_certified':False})
            return 0
        if args.command == "execution-report":
            from .execution_evidence import build_record
            if args.output.exists():raise ConfigurationError('refusing to overwrite execution evidence')
            value=build_record(args.receipt,measurement_path=args.measurement,bundles=args.bundle)
            write_json_atomic(args.output,value)
            _print({'profile_id':value['profile']['profile_id'],'quality_status':value['quality_status'],'output':str(args.output)})
            return 0
        if args.command == "select-configuration":
            from .configuration_select import select
            if args.output.exists():raise ConfigurationError('refusing to overwrite a configuration selection')
            value=select([load_json(p) for p in args.record],load_json(args.budget),load_json(args.policy))
            write_json_atomic(args.output,value);_print(value)
            return 0 if value["selected_profile_id"] is not None else 3
        if args.command == "measure-endpoint":
            from .endpoint_latency import measure
            value=measure(args.trace,args.endpoint,args.receipt,args.output,warmup=args.warmup,iterations=args.iterations)
            _print({k:v for k,v in value.items() if k not in {'samples_ms','action_sha256'}})
            return 0
        if args.command == "realtime-report":
            from .realtime import assess
            if args.output.exists():raise ConfigurationError('refusing to replace a realtime assessment')
            value=assess(load_json(args.measurement),load_json(args.budget))
            write_json_atomic(args.output,value);_print(value)
            return 0
        if args.command == "measure-policy":
            from .latency_probe import measure
            value=measure(args.trace,args.model,args.revision,args.mode,args.output,warmup=args.warmup,iterations=args.iterations,tier_ceiling=args.tier_ceiling)
            _print({k:v for k,v in value.items() if k not in {'samples_ms','action_sha256'}})
            return 0
        if args.command == "checkpoint-view":
            from .checkpoint_view import create_view
            _print(create_view(args.model, args.revision, args.subdir, args.output))
            return 0
        if args.command == "replay-policy":
            from .fixed_input import replay
            result=replay(args.trace,args.endpoint,load_json(args.receipt),args.output,repeats=args.repeats)
            _print({k:v for k,v in result.items() if k!='rows'})
            return 0
        if args.command == "export":
            from .bundle import export_bundle
            _print(export_bundle(args.run, registry, args.output))
            return 0
        if args.command == "verify-bundle":
            from .bundle import verify_bundle
            _print(verify_bundle(args.bundle))
            return 0
        if args.command == "prepare-scenes":
            from .scene_prepare import prepare_scenes
            _print(prepare_scenes(load_json(args.plan), args.output, args.repo_root, workers=args.workers, gpus=args.gpu, reuse_scenes=args.reuse_scenes))
            return 0
        if args.command == "sim-plan":
            from .campaign import build_campaign
            _print(build_campaign(load_json(args.spec), args.output, args.scenes))
            return 0
        if args.command == "simulator-doctor":
            from .simulator_doctor import inspect_simulator
            result = inspect_simulator(args.simulator)
            _print(result)
            return 1 if result["status"] == "blocked_local_renderer" else 0
        if args.command == "coverage":
            from .coverage import coverage
            _print(coverage(registry, args.run))
            return 0
        if args.command == "adapters":
            from .adapters import load_adapters
            _print({"schema_version": 1, "adapters": load_adapters()})
            return 0
        if args.command == "registry":
            payload = {
                "registry_sha256": registry.digest,
                "models": [
                    {
                        "id": model["id"], "revision": model["revision"],
                        "backbone": model["backbone"], "builtin": model["builtin"],
                    }
                    for model in registry.raw["models"]
                ],
                "suites": [suite["id"] for suite in registry.raw["suites"]],
            }
            if args.json:
                _print(payload)
            else:
                print(f"registry {registry.digest}")
                for model in payload["models"]:
                    marker = "built-in" if model["builtin"] else "benchmark checkpoint"
                    print(f"  {model['backbone']:<18} {model['id']}@{model['revision'][:12]}  [{marker}]")
            return 0
        if args.command == "plan":
            arms = load_arms(args.arms)
            value = build_plan(registry, arms, args.profile, _models(args.model))
            write_json_atomic(args.output, value)
            print(f"plan {value['plan_id']} | {value['pair_count']} pairs | {value['job_count']} jobs")
            print(args.output.resolve())
            return 0
        if args.command == "doctor":
            plan = load_json(args.plan)
            value = inspect_plan(plan, registry, require_cached=args.require_cached)
            _print(value)
            return 0 if value["ok"] else 1
        if args.command == "run":
            plan = load_json(args.plan)
            validate_plan(plan)
            value = execute_plan(
                plan, args.output, repo_root=args.repo_root, gpu=args.gpu, fail_fast=args.fail_fast,
                paired_workers=args.paired_workers
            )
            _print(value)
            return 0 if not value["failed"] and value["finished"] else 1
        if args.command == "report":
            value = build_report(
                args.run, registry, allow_synthetic=args.allow_synthetic, output=args.output
            )
            _print(value)
            return 0 if value["reportable"] else 1
        if args.command == "sweep-plan":
            sweep = load_sweep(args.sweep)
            value = build_sweep_plan(registry, sweep, args.profile, _models(args.model))
            write_json_atomic(args.output, value)
            print(
                f"sweep plan {value['plan_id']} | {len(value['sweep']['points'])} schedule "
                f"points vs {value['sweep']['baseline']['id']} | {value['pair_count']} pairs "
                f"| {value['job_count']} jobs"
            )
            print(args.output.resolve())
            return 0
        if args.command == "sweep-report":
            value = build_sweep_report(
                args.run, registry, allow_synthetic=args.allow_synthetic, output=args.output
            )
            print(value["table_markdown"])
            return 0 if value["reportable_as_screening"] else 1
        if args.command == "prefetch":
            return _prefetch(registry, _models(args.model), args.datasets, args.execute)
    except ConfigurationError as error:
        parser.error(str(error))
    return 2


def _prefetch(registry, selected, datasets: bool, execute: bool) -> int:
    model_ids = selected or sorted(registry.builtin_model_ids)
    unknown = sorted(set(model_ids) - set(registry.models))
    if unknown:
        raise ConfigurationError(f"unknown model(s): {', '.join(unknown)}")
    commands = []
    for model_id in model_ids:
        model = registry.models[model_id]
        commands.append(["hf", "download", model["id"], "--revision", model["revision"]])
    if datasets:
        for dataset in registry.raw["datasets"]:
            if dataset["kind"] in {"huggingface_dataset", "simulator"}:
                commands.append(
                    [
                        "hf", "download", dataset["source"], "--type", "dataset",
                        "--revision", dataset["revision"],
                    ]
                )
            for dependency in dataset.get("dependencies", ()):
                if dependency["source"].startswith(("http://", "https://")):
                    print(
                        f"# source checkout must be cloned separately at {dependency['revision']}: "
                        f"{dependency['source']}"
                    )
                    continue
                commands.append(
                    [
                        "hf", "download", dependency["source"], "--type", "dataset",
                        "--revision", dependency["revision"],
                    ]
                )
    unique_commands = []
    seen = set()
    for command in commands:
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            unique_commands.append(command)
    for command in unique_commands:
        print(" ".join(command))
        if execute:
            completed = subprocess.run(command, check=False)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
