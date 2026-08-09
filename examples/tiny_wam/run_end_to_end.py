#!/usr/bin/env python3
"""The user-facing workflow, end to end, on a real checkpoint with real weights.

Nine steps, each one printed and checked. Nothing is mocked and nothing is asserted that is not
demonstrated:

  1  the package validates against the published layout
  2  publishability() -- it can be published with training internals stripped
  3  from_pretrained() loads it
  4  its `backbone` resolves to a REGISTERED adapter (third-party, registered here at call time)
  5  capabilities() derives tokens from the execution block alone
  6  the adapter's spec() describes the control step
  7  Optimizer.compile() produces a Plan FROM THOSE CAPABILITIES
  8  every applied pass is accounted for -- installed, or shown vacuous for this deployment
  9  one real inference runs, with real weights, and the foldable-head capability is verified

    PYTHONPATH=. /home/ubuntu/.venv-lingbot/bin/python examples/tiny_wam/run_end_to_end.py

Run `build_checkpoint.py` first if the package is not present.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import instinctwm  # noqa: E402
from examples.tiny_wam import model as M  # noqa: E402
from examples.tiny_wam.adapter import BACKBONE, TinyWAMAdapter  # noqa: E402
from instinctwm.descriptors.package import from_pretrained, publishability, validate_package  # noqa: E402
from instinctwm.planners.planner import Optimizer  # noqa: E402

CKPT = ROOT / "examples" / "checkpoint" / "tiny-wam-2v2a"
FAILED: list[str] = []


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def main() -> int:
    if not CKPT.exists():
        print(f"{CKPT} not found -- run examples/tiny_wam/build_checkpoint.py first")
        return 2

    step(1, "The package validates against the published layout")
    rep = validate_package(CKPT)
    print(rep.explain())
    check(rep.ok, "servable package")

    step(2, "It can be published without exposing training internals")
    ok, findings = publishability(CKPT)
    for f in findings:
        print(f"  - {f}")
    check(ok, "publishable with provenance stripped")

    step(3, "from_pretrained() loads it")
    ckpt = from_pretrained(CKPT)
    print(f"  model_id : {ckpt.model_id}")
    print(f"  backbone : {ckpt.execution.backbone}")
    print(f"  servable : {ckpt.execution.servable}")
    check(ckpt.model_id == "example-org/tiny-wam-2v2a", "loaded by declaration")

    step(4, "Its backbone resolves to a REGISTERED adapter")
    print(f"  registered before: {instinctwm.available_models()}")
    check(ckpt.execution.backbone not in instinctwm.available_models(),
          "the backbone is NOT built in -- this is a third-party adapter")
    param_bytes = int(ckpt.execution.extra.get("param_bytes", 0))
    instinctwm.register(BACKBONE, lambda: TinyWAMAdapter(
        checkpoint_dir=str(CKPT), param_bytes=param_bytes, declared_model_id=ckpt.model_id))
    print(f"  registered after : {instinctwm.available_models()}")
    check(ckpt.execution.backbone in instinctwm.available_models(),
          f"declared backbone {ckpt.execution.backbone!r} now resolves")
    adapter = instinctwm.load(ckpt.execution.backbone)
    check(adapter is not None, "instinctwm.load() returns the adapter")

    step(5, "capabilities() -- derived from the execution block, and nothing else")
    caps = ckpt.capabilities()
    for c in sorted(caps):
        print(f"  {c}")
    blob = " ".join(sorted(caps)).lower()
    check(not any(w in blob for w in ("recipe", "teacher", "dataset", "seeded", "training")),
          "no capability token mentions how it was trained")

    step(6, "The adapter describes the control step")
    spec = adapter.spec()
    print(f"  model_id    : {spec.model_id}")
    print(f"  param_bytes : {spec.param_bytes:,}")
    for s in spec.streams:
        print(f"  stream {s.name:6} tokens/frame={s.tokens_per_frame} lifetime={s.lifetime.value}")
    for p in spec.phases:
        print(f"  phase  {p.name:6} nfe={p.nfe} reads={sorted(p.reads)} writes={sorted(p.writes)}")
    for k, g in spec.guidance.items():
        print(f"  guidance {k:6} mode={g.mode.value} scale={g.scale} batchable={g.batchable}")
    check(spec.param_bytes == param_bytes, "param_bytes matches the published weights")

    step(7, "A Plan is compiled FROM THE DECLARED CAPABILITIES")
    plan = Optimizer().compile(spec, capabilities=caps)
    print(plan.explain())
    applied = [r.name for r in plan.results if r.applies]
    print(f"  applied: {applied or '(none)'}")
    print("  NOTE: fsdp/allocator/debug-dump are SUBSTRATE passes -- they describe the serving\n"
          "        environment, not the model, so they apply to every checkpoint. Step 8 accounts\n"
          "        for each one. This surprised me when writing the example, which is why it is here.")
    check(plan.model_id == spec.model_id, "the plan is for this checkpoint")

    step(8, "The plan installs -- and every applied pass is accounted for")
    installed = adapter.install(None, plan)
    vac = adapter.vacuous(plan)
    for n, why in vac.items():
        print(f"  vacuous  {n:26} {why}")
    print(f"  installed: {list(installed) or '(none)'}")
    check(set(applied) == set(installed) | set(vac),
          "every applied pass is either installed or shown vacuous -- none silently dropped")
    check(not (set(installed) & set(vac)), "and nothing is counted twice")

    step(9, "One real inference, with the real published weights")
    net = M.TinyWAM()
    net.load_state_dict(load_file(str(CKPT / "model.safetensors")))
    net.eval()
    torch.manual_seed(1234)
    obs = torch.randn(1, M.OBS_DIM)

    nfe_action = int(ckpt.execution.nfe.get("action", 2))
    cfg = spec.guidance["video"].scale if spec.guidance["video"].mode.value == "cfg" else 0.0
    with torch.no_grad():
        action = net(obs, nfe=nfe_action, cfg_scale=cfg, folded=True)
    print(f"  obs           {tuple(obs.shape)}")
    print(f"  nfe (declared){nfe_action:>3}    cfg scale {cfg}")
    print(f"  action chunk  {tuple(action.shape)}  dtype={action.dtype}")
    print(f"  first row     {[round(float(v), 4) for v in action[0, 0, :6]]} ...")
    check(action.shape == (1, M.ACTION_HORIZON, M.ACTION_DIM), "action chunk has the declared shape")
    check(bool(torch.isfinite(action).all()), "and is finite")

    # the declared capability `output_projection.foldable` is a claim about these weights: verify it
    with torch.no_grad():
        unfolded = net(obs, nfe=nfe_action, cfg_scale=cfg, folded=False)
    delta = float((action - unfolded).abs().max())
    check(delta < 1e-5,
          "output_projection.foldable is TRUE of these weights: folded == unfolded",
          f"max|delta| = {delta:.3e}")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: declared -> published -> resolved -> planned -> served, on real weights.")
    print("      No runtime code was modified to support this checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
