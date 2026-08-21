"""`instinctflash` — the command line. Five verbs, no Python required.

    instinctflash devices                    what machine am I on, and what can it do
    instinctflash describe  <model-id>       what a checkpoint declares, without downloading weights
    instinctflash validate  <dir>            is this directory a publishable checkpoint
    instinctflash plan      <model-id>       what the runtime would do to it, and why
    instinctflash run       <model-id>       load it and produce real actions

Why a CLI at all, given `Runtime.from_pretrained` is three lines. Because "install the package, give
it a model id, run" should not require writing a program, and because four of these five verbs answer
questions you want answered BEFORE committing to a download or a GPU: what is this checkpoint, will
this runtime serve it, what would it do to it, and is this machine capable of the plan. `plan` and
`describe` need no weights and no GPU at all.

`run` uses zero-filled observations by default. That is deliberately a smoke test and says so: it
proves this checkpoint loads on this machine and produces finite actions of the right shape, which is
the question a new user actually has. It is not an evaluation and the output is not a result.
"""

from __future__ import annotations

import argparse
import json
import sys


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
    from instinctflash.descriptors.package import publishability, validate_package
    rep = validate_package(a.path)
    print(rep.explain())
    ok = False
    try:
        ok, findings = publishability(a.path)
        print(f"  publishable without training internals: {'YES' if ok else 'NO'}")
        for f in findings:
            print(f"    - {f}")
    except Exception as e:                                       # noqa: BLE001
        print(f"  publishability: {type(e).__name__}: {e}")
    return 0 if rep.ok else 1


def cmd_plan(a) -> int:
    from instinctflash import Runtime
    try:
        rt = Runtime.from_pretrained(a.model, strict=not a.any_checkpoint)
    except Exception as e:                                       # noqa: BLE001
        print(f"{a.model}: {type(e).__name__}: {e}")
        return 1
    print(rt.explain())
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

    print(f"SMOKE TEST -- zero-filled observations. This proves the checkpoint loads here and "
          f"returns finite actions.\nIt is not an evaluation.\n")
    times, last = [], None
    with rt, rt.episode(**({"prompt": a.prompt} if a.prompt else {})) as ep:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="instinctflash", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("devices", help="what machine am I on").set_defaults(fn=cmd_devices)

    d = sub.add_parser("describe", help="what a checkpoint declares, without its weights")
    d.add_argument("model")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=cmd_describe)

    v = sub.add_parser("validate", help="is this directory a publishable checkpoint")
    v.add_argument("path")
    v.set_defaults(fn=cmd_validate)

    p = sub.add_parser("plan", help="what the runtime would do, and why")
    p.add_argument("model")
    p.add_argument("--any-checkpoint", action="store_true",
                   help="do not refuse a checkpoint declaring servable=false")
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
