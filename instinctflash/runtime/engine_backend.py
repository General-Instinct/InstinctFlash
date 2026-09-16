"""EngineBackend — execute a checkpoint through the fused serving engine in `serving/`.

An ExecutionBackend, deliberately NOT an adapter: the adapter says WHAT the model is (that does
not change with hardware), the planner says what is valid, and this class is only a different
answer to WHERE/HOW the same checkpoint runs. The current route requires an
explicit FP8 request and SM110. Performance must be measured at a matched action
horizon and input/output boundary; older pi05 timing ratios mixed horizons.

Accuracy stance: FP8 W+A changes numerics. Historical dedicated-server quality
results do not certify the generic Runtime's current state, camera and action
buffering contract. See eval/fp8_comparison_2026-09-09/README.md for the audit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: Backbones the engine ships a frontend for AND this repo has driven end-to-end. Widening this
#: list is T2 work: a new frontend plus its own parity gates, never just a name added here.
ENGINE_BACKBONES = ("pi05", "lingbot_vla", "lingbot_vla_v2", "groot_n17", "wan_va", "cosmos3_policy", "dreamzero")

# Legacy standalone defaults; Runtime uses checkpoint geometry and native decoding.
PI05_ENGINE_ACTION_DIM = 7
PI05_ENGINE_ACTION_CHUNK = 10
PI05_ENGINE_MAX_ACTION_DIM = 32

#: The engine's pi05 frontend DENOISE SCHEDULE as built. Not a default the frontend reads at
#: infer time — a number baked at weight load: ``action_out_proj`` is pre-scaled by ``-1/steps``
#: (pi05_thor.py ``steps = 10`` at weight load), the per-step AdaRMSNorm time tables are sized
#: and filled for exactly ``steps`` values of t (``steps = 10`` in ``set_prompt``), and every
#: captured decoder graph and calibration pass receives ``'steps': 10`` in its dims dict (six
#: sites). ``Runtime.from_pretrained(..., nfe={"action": 1}, placement="engine")`` therefore
#: used to ACCEPT the request and run 11 forwards: the plan said ``action=1``, ``explain()``
#: said 2 forwards, the engine ran the 10-step graph and returned the nfe10 digest (Thor probe
#: 2026-09-02, iwm_distill/fewstep/cert_pi05_nfe1_thor.md §2). That is the geometry gate's
#: failure class on a different axis — a declared operating point silently not honored — and
#: it is closed the same way: the engine tier serves its baked operating point or DECLINES.
PI05_ENGINE_ACTION_STEPS = 10


@dataclass(frozen=True)
class EngineBakedOperatingPoint:
    """What one engine frontend BAKES about the operating point, declared per engine build.

    An operating point is the tuple (schedule grid, per-stream guidance scale, CFG batching)
    (docs/rfc/fewstep-distillation.md §11). A fused engine does not read that tuple at infer
    time: the step count sizes its time tables, scales its output projection and rides inside
    its captured graphs; guidance is a pipeline it either builds or does not. So each frontend
    serves exactly ONE operating point, and the runtime must know which — a request for any
    other point is declined at plan time (``passes/generic/engine_offload.py``) and refused at
    build time (``EngineBackend``), never served as something else.

    ``steps_parameterized`` is False for every frontend in ``serving/`` today, and it may only
    become True together with a test that PROVES the frontend serves the requested count
    (digest differs from the baked count's, per-step tables rebuilt, graphs recaptured) — never
    because a constructor grew a parameter. The experimental step-parameterized pi05 scratch
    build on Thor (``patch_pi05_thor_steps.py``, 30.6 ms at nfe1) is exactly that unproven
    case and stays outside this table until it lands in ``serving/`` with a certificate.
    """

    #: the backbone id a checkpoint declares; None for a frontend this repo ships no adapter for
    backbone: "str | None"
    #: the frontend module under serving/, so the mirror test can read the source it mirrors
    frontend: str
    #: per denoise stream (the adapter's PhaseSpec name), the step count the build bakes
    steps: Mapping[str, int]
    #: per stream, the (mode, scale) the build SERVES — as built by EngineBackend, not what the
    #: frontend could be switched to (pi05_thor has an RL-mode CFG pipeline behind
    #: ``set_rl_mode``; EngineBackend never enables it, so the served point is no CFG)
    guidance: Mapping[str, tuple[str, float]]
    #: regexes with ONE capture group each; applied to the frontend source, the set of captured
    #: integers must equal {steps} — the CI mirror check and the build-time live check
    source_patterns: tuple[str, ...]
    #: where in the frontend the count is baked, in words a decline reason can quote
    evidence: str
    steps_parameterized: bool = False

    def served_point(self) -> str:
        steps = ", ".join(f"{s}={n} steps" for s, n in sorted(self.steps.items()))
        guidance = ", ".join(f"{s}={m}@{w:g}" for s, (m, w) in sorted(self.guidance.items()))
        return f"{steps}; guidance {guidance or 'none'}"


ENGINE_BAKED_OPERATING_POINTS: "tuple[EngineBakedOperatingPoint, ...]" = (
    EngineBakedOperatingPoint(
        backbone="pi05",
        frontend="serving/flash_rt/frontends/torch/pi05_thor.py",
        steps={"action": PI05_ENGINE_ACTION_STEPS},
        guidance={"action": ("none", 1.0)},
        source_patterns=(r"(?m)^\s*steps\s*=\s*(\d+)\b", r"['\"]steps['\"]\s*:\s*(\d+)\b"),
        evidence=("action_out_proj is pre-scaled by -1/10 at weight load, the per-step "
                  "AdaRMSNorm time tables are sized and filled for 10 values of t, and every "
                  "captured decoder graph carries 'steps': 10"),
    ),
    EngineBakedOperatingPoint(
        backbone="lingbot_vla",
        frontend="serving/flash_rt/frontends/torch/vla4b_thor.py",
        steps={"action": 10},
        guidance={"action": ("none", 1.0)},
        source_patterns=(r"(?m)^STEPS\s*=\s*(\d+)\b",),
        evidence=("the module constant STEPS = 10 sizes the AdaRMS step tables "
                  "(build_step_tables(steps=STEPS)) and rides in the pipeline dims dict"),
    ),
    EngineBakedOperatingPoint(
        backbone="lingbot_vla_v2",
        frontend="serving/flash_rt/frontends/torch/vla2_thor.py",
        steps={"action": 10},
        guidance={"action": ("none", 1.0)},
        source_patterns=(r"(?m)^STEPS\s*=\s*(\d+)\b",),
        evidence=("the module constant STEPS = 10 sizes the AdaRMS step tables "
                  "(build_step_tables(steps=STEPS)) and rides in the pipeline dims dict"),
    ),
    EngineBakedOperatingPoint(
        backbone="groot_n17",
        frontend="serving/flash_rt/frontends/torch/groot_n17_thor.py",
        steps={"action": 4},
        guidance={"action": ("none", 1.0)},
        source_patterns=(r"num_inference_timesteps:\s*int\s*=\s*(\d+)\b",),
        evidence=("infer(num_inference_timesteps=4) is a parameter in name only on the served "
                  "path: set_prompt's warmup captures one DiT graph per step at the default 4 "
                  "and the per-step AdaLN modulators are computed once; any other count drops "
                  "to the uncaptured eager loop, which nothing certifies"),
    ),
    EngineBakedOperatingPoint(
        backbone=None,   # GROOT N1.6: a frontend ships, no adapter in this repo declares it
        frontend="serving/flash_rt/frontends/torch/groot_thor.py",
        steps={"action": 4},
        guidance={"action": ("none", 1.0)},
        source_patterns=(r"self\.num_steps\s*=\s*(\d+)\b",),
        evidence=("self.num_steps = 4 sizes the precomputed timestep embeddings and the whole "
                  "4-step DiT loop is captured in one CUDA graph"),
    ),
)


#: Backbones whose engine declares its operating point PER BUILD rather than as a source
#: literal. The wan_va (LingBot-VA) Thor engine is constructed FOR a ``WanVaOperatingPoint``
#: (serving/flash_rt/models/wan_va/operating_point.py): the build sizes its AdaLN step tables
#: from the declared grids, allocates a 1- or 2-stream KV slab from the declared guidance and
#: runs (V+1)+(A+1)+2 DiT forwards per cycle — so there is no literal in the frontend source to
#: mirror, and an UNBUILT wan_va engine has no point at all. A built instance publishes
#: ``declaration()``; ``wan_va_baked_operating_point`` turns it into the same
#: ``EngineBakedOperatingPoint`` record every other engine has, with ``steps_parameterized``
#: False — the INSTANCE bakes its point exactly like pi05_thor bakes 10 (the build-time
#: parameter is not a serve-time switch). Proven by tests/test_wan_va_engine_operating_point.py
#: (tables/forward counts differ per point; the serving-side mismatch rule and this one agree on
#: every request) and by the Thor M1 record (distinct action digests per point).
BUILD_DECLARED_BACKBONES = ("wan_va", "cosmos3_policy", "dreamzero")


def wan_va_baked_operating_point(declaration: Mapping) -> EngineBakedOperatingPoint:
    """The record for ONE built wan_va engine, from ``WanVaTorchFrontendThor.declaration()``
    (or ``WanVaOperatingPoint.declaration()``). No source pattern: the count is not a literal."""
    steps = {str(k): int(v) for k, v in dict(declaration["steps"]).items()}
    guidance = {str(k): (str(m), float(w)) for k, (m, w) in dict(declaration["guidance"]).items()}
    return EngineBakedOperatingPoint(
        backbone="wan_va",
        frontend=str(declaration.get("frontend", "serving/flash_rt/frontends/torch/wan_va_thor.py")),
        steps=steps, guidance=guidance, source_patterns=(),
        evidence=str(declaration.get("evidence", "declared by the build")),
        steps_parameterized=False)


def engine_baked_operating_point(backbone, build=None) -> "EngineBakedOperatingPoint | None":
    """The engine build's declared operating point for this backbone, or None when no
    frontend in ``serving/`` is declared for it (choose_backend never builds the engine then).

    ``build`` — for a build-declared backbone (``BUILD_DECLARED_BACKBONES``), the built
    engine's ``declaration()`` dict; without it such a backbone has NO point (nothing is built,
    nothing can be served) and None is returned, which keeps the engine off for it."""
    if backbone in BUILD_DECLARED_BACKBONES:
        if build is None:
            return None
        if backbone == "wan_va":
            return wan_va_baked_operating_point(build)
        if backbone in ("cosmos3_policy", "dreamzero"):
            return EngineBakedOperatingPoint(
                backbone=backbone, frontend=str(build["frontend"]),
                steps=dict(build["steps"]), guidance=dict(build["guidance"]),
                source_patterns=(), evidence=str(build["evidence"]))
    for cap in ENGINE_BAKED_OPERATING_POINTS:
        if cap.backbone is not None and cap.backbone == backbone:
            return cap
    return None


def baked_step_literals(source: str, patterns) -> "set[int]":
    """Every step-count literal the patterns find in a frontend's source. The mirror is honest
    only while this set is exactly {the declared count}: an empty set means the literal moved
    (or became a parameter), two values mean the build is not one operating point."""
    import re
    found: set[int] = set()
    for pat in patterns:
        found.update(int(m) for m in re.findall(pat, source))
    return found


def _served_guidance(mode: str, scale) -> "tuple[bool, float]":
    """The served semantics of a (mode, scale): does a negative branch get computed and
    combined, and at what scale. ``none``, ``positive_only`` and ``cfg@1`` all serve the
    positive branch alone and are the SAME computation (descriptors/guidance.py)."""
    negative = mode == "cfg" and (scale is None or float(scale) > 1.0)
    return negative, (float(scale) if negative and scale is not None else 1.0)


def engine_operating_point_problem(
    baked: EngineBakedOperatingPoint,
    requested_steps: Mapping[str, "int | None"],
    requested_guidance: "Mapping[str, tuple[str, float | None] | None]",
    *,
    requester: str = "this plan's operating point",
) -> "str | None":
    """Why the engine build cannot serve the requested operating point — ONE rule, two surfaces.

    Called by the planner pass with the spec's phases/guidance (plan time) and by
    ``EngineBackend`` with the checkpoint's declaration plus the ``nfe=`` override (build time),
    so the two cannot disagree. Returns the decline reason, or None when the request IS the
    baked point. A stream whose request cannot be read is UNVERIFIED and declines — a
    schedule-baked pipeline is never run blind.
    """
    fam = baked.backbone or baked.frontend
    for stream, n in sorted(baked.steps.items()):
        req = requested_steps.get(stream)
        try:
            req = int(req) if req is not None else None
        except (TypeError, ValueError):
            req = None
        if req is None:
            return (f"engine {fam} frontend bakes a {n}-step {stream} schedule as built "
                    f"({baked.evidence}), and the {stream} step count {requester} requests "
                    f"could not be verified. Declining rather than running a schedule-baked "
                    f"pipeline blind — the torch placement serves the declared schedule.")
        if req != n and not baked.steps_parameterized:
            return (f"engine {fam} frontend bakes a {n}-step {stream} schedule as built "
                    f"({baked.evidence}); {requester} requests {stream}={req} — falling back "
                    f"to the torch placement, which serves the declared schedule. The engine "
                    f"tier honors the operating point or declines; it never silently runs "
                    f"{n} steps under a plan that says {req}.")
    for stream, (mode, scale) in sorted(baked.guidance.items()):
        req = requested_guidance.get(stream)
        if req is None:
            return (f"engine {fam} frontend serves {stream} guidance {mode}@{scale:g} as built, "
                    f"and the {stream} guidance {requester} requests could not be verified. "
                    f"Declining rather than guessing — the torch placement serves the "
                    f"declared guidance.")
        want_neg, want_scale = _served_guidance(*req)
        have_neg, have_scale = _served_guidance(mode, scale)
        if want_neg != have_neg or (want_neg and want_scale != have_scale):
            want = f"{req[0]}@{'?' if req[1] is None else format(float(req[1]), 'g')}"
            return (f"engine {fam} frontend serves {stream} guidance {mode}@{scale:g} as built "
                    f"(no negative branch is computed or combined; the frontend's CFG pipeline "
                    f"is not enabled by this backend); {requester} requests {stream}={want} — "
                    f"falling back to the torch placement, which serves the declared guidance. "
                    f"The engine tier honors the operating point or declines.")
    return None


def requested_operating_point(adapter, checkpoint, nfe=None) -> "tuple[dict, dict, str]":
    """The operating point a build request asks for: (steps per stream, (mode, scale) per
    stream, where it came from). The same resolution the planner performs in
    ``facade._compile_declaration`` — family spec, then ``execution.nfe`` / ``execution.guidance``
    from the declaration, then the caller's ``nfe=`` override — so build time reads the tuple
    the plan was priced at. Without an adapter the declaration and the override are all there is;
    a stream neither names stays unresolved and the caller declines on it.
    """
    from instinctflash.descriptors.guidance import resolve

    execution = getattr(checkpoint, "execution", None)
    declared_nfe = dict(getattr(execution, "nfe", None) or {})
    declared_guidance = dict(getattr(execution, "guidance", None) or {})
    override = dict(nfe or {})
    schedule = {**declared_nfe, **override}
    sources = []
    if override:
        sources.append("the nfe= override")
    if declared_nfe:
        sources.append("the checkpoint's execution.nfe")

    spec = None
    hook = getattr(adapter, "spec_for_checkpoint", None)
    if callable(hook):
        spec = hook(checkpoint)
    elif callable(getattr(adapter, "spec", None)):
        spec = adapter.spec()
    if spec is not None:
        if schedule:
            spec = spec.with_nfe(schedule)
        spec = spec.with_guidance(declared_guidance)
        steps = {p.name: int(p.nfe) for p in spec.phases}
        guidance = {name: (rule.mode.value, float(rule.scale))
                    for name, rule in spec.guidance.items()}
        sources.append("the family adapter's declared schedule")
    else:
        steps = {k: int(v) for k, v in schedule.items()}
        guidance = {s: (r.mode, r.scale) for s, r in resolve(declared_guidance).items()}
    return steps, guidance, " over ".join(sources) or "no declaration and no override"


def declared_action_dim(checkpoint) -> "tuple[int | None, str]":
    """The action dimensionality this checkpoint declares, and where the answer came from.

    Two sources, in trust order — both are the checkpoint speaking, never a guess:

      1. ``execution.action_dim`` in the declaration (the Cosmos3 entries already use this key).
      2. The checkpoint's own ``config.json`` ``output_features.action.shape`` — the lerobot
         policy releases all carry it (base declares [32], the LIBERO v044 fine-tune [7]), and
         pointer packages get it too because ``_declared_view`` symlinks the snapshot's files.

    Returns ``(None, reason)`` when neither source answers. Callers must treat that as
    UNVERIFIED and keep the engine off — running a geometry-baked pipeline blind is exactly the
    failure mode this helper exists to close.
    """
    extra = getattr(checkpoint.execution, "extra", None) or {}
    v = extra.get("action_dim")
    if v is not None:
        try:
            return int(v), "the declaration's execution.action_dim"
        except (TypeError, ValueError):
            return None, f"execution.action_dim is not an integer ({v!r})"
    cfg_path = Path(str(checkpoint.path)) / "config.json"
    if cfg_path.is_file():
        import json
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, ValueError) as e:
            return None, f"config.json next to the weights is unreadable ({type(e).__name__})"
        shape = ((cfg.get("output_features") or {}).get("action") or {}).get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) == 1:
            try:
                return int(shape[0]), "the checkpoint's own config.json output_features.action.shape"
            except (TypeError, ValueError):
                pass
        return None, ("config.json is present but declares no 1-D "
                      "output_features.action.shape to read the action dimensionality from")
    return None, ("no execution.action_dim in the declaration and no config.json next to the "
                  "checkpoint to read output_features.action.shape from")


def engine_available(backbone=None) -> "tuple[bool, str]":
    """Can this interpreter run the engine pipeline? Returns (ok, reason) — the reason is what
    `explain()` shows when the answer is no, so it names the actual missing piece."""
    try:
        import torch
    except ImportError:
        return False, "no torch in this interpreter"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    cap = torch.cuda.get_device_capability()
    if cap == (8, 9):
        from .sm89_fp8 import available
        return available(backbone)
    if cap == (12, 0):
        from .sm120_fp8 import available
        return available(backbone)
    if cap == (9, 0):
        return (hasattr(torch, "_scaled_mm"), "H100 PyTorch E4M3 projection executor requires torch._scaled_mm")
    if cap != (11, 0):
        return False, (f"device is SM{cap[0]}{cap[1]}, engine kernels ship for SM110 "
                       f"(thor); the torch chain is also the measured winner off-Thor")
    if backbone in ("cosmos3_policy", "dreamzero"):
        if not hasattr(torch, "_scaled_mm"):
            return False, f"{backbone} FP8 requires torch._scaled_mm"
        return True, "SM110 + PyTorch FP8 projections; native policy processing"
    try:
        import flash_rt  # noqa: F401
    except ImportError as e:
        return False, f"flash_rt not importable here ({e}); build serving/ on this device first"
    try:
        from importlib import import_module
        kernels = import_module("flash_rt.flash_rt_kernels")
    except (ImportError, OSError) as e:
        return False, (f"flash_rt compiled kernels cannot load ({e}); build serving/ "
                       "for this Thor interpreter and its CUDA environment")
    missing = [name for name in ("GemmRunner", "FvkContext")
               if not callable(getattr(kernels, name, None))]
    if missing:
        return False, f"flash_rt compiled kernels lack required exports {missing}; rebuild serving/"
    return True, "SM110 + engine kernels present"


class EngineBackend:
    """Serve supported families through Thor engines and native robot processors."""

    #: default prompt for reset()-less smoke paths; a real episode always passes its own.
    _SMOKE_PROMPT = "smoke test: reach forward"

    def __init__(self, adapter, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
        backbone = checkpoint.execution.backbone
        if backbone not in ENGINE_BACKBONES:
            raise RuntimeError(
                f"engine backend has no frontend for backbone {backbone!r} "
                f"(supported: {ENGINE_BACKBONES}). This is a T2 gap, not a configuration error.")
        import torch
        if any(r.name == "engine_offload" and r.applies and r.params.get("executor") == "sm120_torch_fp8"
               for r in getattr(plan, "results", ())):
            from .sm120_fp8 import build_sm120_loop
            self._loop = build_sm120_loop(
                adapter, checkpoint, plan, device=device, nfe=nfe, step_cache=step_cache)
            self._checkpoint = checkpoint
            return
        if any(r.name == "engine_offload" and r.applies and r.params.get("executor") == "sm89_torch_fp8"
               for r in getattr(plan, "results", ())):
            from .sm89_fp8 import build_sm89_loop
            self._loop = build_sm89_loop(
                adapter, checkpoint, plan, device=device, nfe=nfe, step_cache=step_cache)
            self._checkpoint = checkpoint
            return
        if any(r.name == "engine_offload" and r.applies and r.params.get("executor") == "h100_torch_fp8"
               for r in getattr(plan, "results", ())):
            if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (9, 0):
                raise RuntimeError("H100 FP8 plan requires an SM90 device")
            from instinctflash.runtime.h100_fp8 import build_h100_loop
            self._loop = build_h100_loop(
                adapter, checkpoint, plan, device=device, nfe=nfe,
                **({"step_cache": step_cache} if step_cache is not None else {}))
            self._checkpoint = checkpoint
            return
        if backbone in ("cosmos3_policy", "dreamzero"):
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (11, 0):
                raise RuntimeError(f"{backbone} FP8 requires a qualified Thor SM110 device")
            steps, guidance, source = requested_operating_point(adapter, checkpoint, nfe)
            build_kw = {"plan": plan} if backbone == "dreamzero" else {}
            if backbone == "dreamzero" and step_cache is not None:
                build_kw["step_cache"] = step_cache
            loop = adapter.build_fp8(checkpoint, device=device, nfe=nfe, **build_kw)
            try:
                declaration = loop.declaration()
                stats = loop.backend_stats
                if callable(stats):
                    stats = stats()
                receipt = stats.get("fp8_recipe") or {}
                if declaration.get("precision") != "fp8" or not receipt.get("projections"):
                    raise RuntimeError(f"{backbone} FP8 build did not install actual FP8 projections")
                baked = engine_baked_operating_point(backbone, declaration)
                problem = engine_operating_point_problem(
                    baked, steps, guidance, requester=f"the requested operating point ({source})")
                if problem:
                    raise RuntimeError(f"engine placement refused: {problem}")
            except Exception:
                loop.close()
                raise
            self._loop, self._checkpoint = loop, checkpoint
            return
        if backbone == "wan_va":
            ok, why = engine_available()
            if not ok:
                raise RuntimeError(f"engine backend unavailable: {why}")
            from instinctflash.runtime.wan_va_engine_build import build_wan_va_engine_loop
            steps, guidance, source = requested_operating_point(adapter, checkpoint, nfe)
            loop = build_wan_va_engine_loop(adapter, checkpoint, device=device, nfe=nfe)
            try:
                baked = wan_va_baked_operating_point(loop.declaration())
                problem = engine_operating_point_problem(
                    baked, steps, guidance, requester=f"the requested operating point ({source})")
                if problem is not None:
                    raise RuntimeError(f"engine placement refused: {problem}")
            except Exception:
                loop.close()
                raise
            self._loop, self._checkpoint = loop, checkpoint
            return
        dim = None
        if backbone == "pi05":
            dim, dim_src = declared_action_dim(checkpoint)
            if dim is None:
                raise RuntimeError(f"engine placement refused: action geometry cannot be verified ({dim_src})")
            if not 1 <= dim <= PI05_ENGINE_MAX_ACTION_DIM:
                raise RuntimeError(
                    f"engine supports action_dim=1..{PI05_ENGINE_MAX_ACTION_DIM}; checkpoint declares {dim}. "
                    "Refusing rather than truncating the declared actions.")
        # OPERATING-POINT GATE, same shape as the geometry gate and for the same reason: the
        # engine bakes its denoise schedule (and serves no CFG), so a request for any other
        # (nfe, guidance) must be refused here, not accepted and served as the baked one. The
        # planner's engine_offload pass declines it at plan time; placement='engine' builds this
        # backend regardless of the plan, and a stale plan could disagree with the nfe= that
        # actually arrives — so the rule runs again on what THIS build was asked to serve.
        # Before this gate: nfe={"action": 1} at placement='engine' returned the nfe10 digest.
        baked = engine_baked_operating_point(backbone)
        if baked is None:
            raise RuntimeError(
                f"engine placement refused: no engine build declares its baked operating point "
                f"for backbone {backbone!r} (ENGINE_BAKED_OPERATING_POINTS). A schedule-baked "
                f"pipeline is never run blind — declare the frontend's steps and guidance, or "
                f"serve the torch chain (placement='in_process'/'auto').")
        steps, guidance, source = requested_operating_point(adapter, checkpoint, nfe)
        problem = engine_operating_point_problem(
            baked, steps, guidance,
            requester=f"the requested operating point (from {source})")
        if problem is not None:
            raise RuntimeError(
                f"engine placement refused: {problem} Serve the torch chain instead "
                f"(placement='in_process' or 'auto'), which honors nfe= and the declared "
                f"guidance; the engine's certificates cover its baked point only.")
        ok, why = engine_available()
        if not ok:
            raise RuntimeError(f"engine backend unavailable: {why}")

        if torch.cuda.get_device_capability(device) != (11, 0):
            raise RuntimeError("engine backend unavailable: Thor fused frontend requires SM110; use an H100 FP8 plan on SM90")

        if backbone == "lingbot_vla":
            from flash_rt.frontends.torch.vla4b_thor import Vla4bTorchFrontendThor
            frontend = Vla4bTorchFrontendThor
        elif backbone == "lingbot_vla_v2":
            from flash_rt.frontends.torch.vla2_thor import Vla2TorchFrontendThor
            frontend = Vla2TorchFrontendThor
        elif backbone == "groot_n17":
            from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
            frontend = GrootN17TorchFrontendThor
        else:
            from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
            frontend = Pi05TorchFrontendThor
            if getattr(frontend, "MODEL_ACTION_DIM", None) != PI05_ENGINE_MAX_ACTION_DIM:
                raise RuntimeError("installed pi05 engine does not expose the required normalized action geometry")
        import inspect
        import sys
        try:
            live_src = inspect.getsource(sys.modules[frontend.__module__])
        except (OSError, TypeError, KeyError) as e:
            raise RuntimeError(f"engine schedule mirror cannot be verified for {backbone}: {e}") from e
        live_steps = baked_step_literals(live_src, baked.source_patterns)
        if live_steps != set(baked.steps.values()):
            raise RuntimeError(
                f"engine schedule mirror is stale: live {backbone} literals {sorted(live_steps)} "
                f"disagree with the declared {sorted(set(baked.steps.values()))}")

        ckpt_dir = Path(checkpoint.path)
        # V2 and GR00T resolve/validate their sharded checkpoints in the family
        # builders. A root-level single-file check is inapplicable.
        if backbone not in {"lingbot_vla_v2", "groot_n17"} and not (ckpt_dir / "model.safetensors").exists():
            raise RuntimeError(f"{ckpt_dir} has no model.safetensors — the engine frontend "
                               f"loads safetensors checkpoints only")
        if backbone == "lingbot_vla":
            from instinctflash.runtime.vla4_engine import build_vla4_engine_loop
            self._loop = build_vla4_engine_loop(checkpoint, device=device, action_steps=steps["action"])
        elif backbone == "lingbot_vla_v2":
            from instinctflash.runtime.vla2_engine import build_vla2_engine_loop
            self._loop = build_vla2_engine_loop(checkpoint, device=device, action_steps=steps["action"])
        elif backbone == "groot_n17":
            from instinctflash.runtime.groot_engine import build_groot_engine_loop
            self._loop = build_groot_engine_loop(checkpoint, device=device, action_steps=steps["action"])
        else:
            from instinctflash.runtime.pi05_engine import Pi05EngineLoop
            self._loop = Pi05EngineLoop.from_checkpoint(checkpoint, device=device, action_steps=steps["action"])
            if self._loop._action_dim != dim:
                self._loop.close()
                raise RuntimeError("pi05 declaration action dimension disagrees with checkpoint config")
        self._checkpoint = checkpoint

    def reset(self, **conditioning: Any) -> None:
        self._loop.reset(**conditioning)

    def predict(self, observation: Mapping[str, Any], *, executed_action: Any = None) -> Any:
        return self._loop.predict(observation, executed_action=executed_action)

    def close(self) -> None:
        if self._loop is not None:
            self._loop.close()
            self._loop = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def _map_observation(observation: Mapping[str, Any]) -> dict:
    """Accept both the engine's native keys and lerobot-style keys.

    Engine wants {'image', 'wrist_image'[, 'wrist_image_right']} as (224,224,3) uint8/fp16.
    lerobot pi05 observations carry 'observation.images.<cam>'. The base camera maps to
    'image'; wrist cameras in declaration order fill the rest.
    """
    if "image" in observation or "images" in observation:
        return dict(observation)
    imgs = [(k, v) for k, v in observation.items() if k.startswith("observation.images.")]
    if not imgs:
        raise ValueError(
            f"observation has neither engine keys ('image'/'images') nor lerobot keys "
            f"('observation.images.*'); got {sorted(observation)[:6]}")
    imgs.sort(key=lambda kv: (0 if "base" in kv[0] or "high" in kv[0] or "top" in kv[0] else 1,
                              kv[0]))
    out = {"image": _hwc(imgs[0][1])}
    if len(imgs) > 1:
        out["wrist_image"] = _hwc(imgs[1][1])
    if len(imgs) > 2:
        out["wrist_image_right"] = _hwc(imgs[2][1])
    return out


def _hwc(img):
    import numpy as np
    a = img
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = a.transpose(1, 2, 0)                                  # CHW -> HWC
    # The frontend accepts exactly two encodings: uint8 [0,255], or fp16 already normalized to
    # [-1,1]. Non-negative floats are pixel data ([0,1] or [0,255]) and become uint8; a float
    # with negative values is already [-1,1]-normalized and must pass through as fp16 --
    # clipping it into [0,255] would silently destroy the signal, not convert it.
    if a.dtype != "uint8":
        mn, mx = float(a.min()), float(a.max())
        if mn >= 0.0:
            scale = 255.0 if mx <= 1.001 else 1.0
            a = (a * scale).clip(0, 255).astype("uint8")
        else:
            a = a.astype("float16")
    return a
