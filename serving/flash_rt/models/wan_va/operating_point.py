"""The operating point a wan_va engine BUILD is made for — declared, derived, enforced.

An operating point is the tuple (per-stream schedule grid, per-stream guidance (mode, scale),
CFG batching) — docs/rfc/fewstep-distillation.md §11 and
``instinctflash/runtime/engine_backend.py::EngineBakedOperatingPoint``. The wan_va engine is
built FOR one such point: the AdaLN step tables are sized from the two grids, the KV slab's
stream count from the guidance (a negative branch means two genuinely different KV streams),
the CFG combine exists or does not, and the forward count per control cycle follows. Nothing
here is read at infer time to "switch" points — the build IS the point, and the frontend
refuses to serve any other (the 9bf2337 discipline: a declared fact is honored or declined,
never silently substituted; the pi05 engine had served its 10-step graph under a plan that
said 2 forwards).

Two points are first-class in Stage 2 (h2_thor_realtime_design.md §5.1):

    POINT_2V4A_W5 — the CERTIFIED point (n=1153 paired, Δ −1.65 pp): video 2 steps @ cfg w=5,
                    action 4 steps positive-only, CFG batch 2, 10 DiT forwards per cycle.
    POINT_2V2A_W1 — the guidance-off point (screen +0.9 pp, certificate in flight): video 2 steps
                    positive-only, action 2 steps positive-only, batch 1, 8 forwards per cycle.

Pure Python + the step_tables schedule math; import-clean everywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from flash_rt.models.wan_va.step_tables import euler_deltas, forward_t_values

STREAMS = ("video", "action")
SHIFT = {"video": 5.0, "action": 1.0}          # snr_shift / action_snr_shift (va_robotwin_cfg)
MODES = ("cfg", "positive_only", "none")       # descriptors/guidance.py vocabulary
_NO_NEGATIVE = ("positive_only", "none")


class OperatingPointMismatch(RuntimeError):
    """The engine was asked to serve a point it was not built for. Fail-closed."""


def _served_guidance(mode: str, scale: Optional[float]) -> tuple[bool, float]:
    """(negative branch computed?, served scale) — ``none``, ``positive_only`` and ``cfg@1``
    are the SAME computation (mirrors engine_backend._served_guidance)."""
    negative = mode == "cfg" and (scale is None or float(scale) > 1.0)
    return negative, (float(scale) if negative and scale is not None else 1.0)


@dataclass(frozen=True)
class WanVaOperatingPoint:
    video_steps: int
    action_steps: int
    video_guidance: tuple = ("cfg", 5.0)
    action_guidance: tuple = ("positive_only", 1.0)
    video_shift: float = 5.0
    action_shift: float = 1.0

    def __post_init__(self):
        for name in ("video_shift", "action_shift"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
            object.__setattr__(self, name, value)
        for name, n in (("video_steps", self.video_steps),
                        ("action_steps", self.action_steps)):
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"{name} must be an int >= 1, got {n!r}")
        for name, g in (("video_guidance", self.video_guidance),
                        ("action_guidance", self.action_guidance)):
            mode, scale = g
            if mode not in MODES:
                raise ValueError(f"{name}: mode {mode!r} not in {MODES}")
            object.__setattr__(self, name, (str(mode), float(scale)))

    # ── the declared tuple ──────────────────────────────────────────
    @property
    def steps(self) -> dict:
        return {"video": self.video_steps, "action": self.action_steps}

    @property
    def guidance(self) -> dict:
        return {"video": self.video_guidance, "action": self.action_guidance}

    def negative_branch(self, stream: str) -> bool:
        return _served_guidance(*self.guidance[stream])[0]

    @property
    def cfg_batch(self) -> int:
        """2 when ANY stream computes a negative branch. The stock server duplicates the batch
        on EVERY forward once ``use_cfg`` is set (server:_repeat_input_for_cfg) and allocates a
        two-stream pool — the streams' K/V genuinely differ (cross-attn conditioning differs
        from layer 0), so the engine slab must hold both."""
        return 2 if any(self.negative_branch(s) for s in STREAMS) else 1

    def combine_scale(self, stream: str) -> float:
        """The w applied by the CFG combine on this stream (1.0 = positive-only, no combine)."""
        return _served_guidance(*self.guidance[stream])[1]

    # ── derived schedule facts ─────────────────────────────────────
    @property
    def shifts(self) -> dict:
        return {"video": self.video_shift, "action": self.action_shift}

    def t_values(self, stream: str) -> list:
        """Padded forward t list: n scheduler timesteps + the trailing 0.0 pred-commit forward."""
        return forward_t_values(self.steps[stream], self.shifts[stream])

    def euler(self, stream: str) -> list:
        return euler_deltas(self.steps[stream], self.shifts[stream])

    def t0_index(self, stream: str) -> int:
        """Index of the t=0 row in the stream's table (the pred-commit AND kv-commit row)."""
        return len(self.t_values(stream)) - 1

    @property
    def forwards_per_cycle(self) -> int:
        """(V+1) video + (A+1) action + 2 kv-commit DiT forwards (mapping_memo §0.12) — the STOCK
        count. The served count with P010 (``forwards_per_cycle_served``) drops the action
        pred-commit forward."""
        return (self.video_steps + 1) + (self.action_steps + 1) + 2

    def forwards_per_cycle_served(self, action_terminal_elision: bool = True) -> int:
        """P010 ``action_terminal_forward_elision`` (InstinctFlash d7e6103, BITEXACT through the
        ring wrap, in shipped_configuration): the ACTION pred-commit forward (t=0, update_cache=1)
        has no reader — its output is discarded and its provisional K/V is dropped by
        clear_pred_cache before any forward runs — so the engine skips the compute and replays
        only its slab allocation (eviction side effect preserved). 9 at 2V/4A, 7 at 2V/2A."""
        return self.forwards_per_cycle - (1 if action_terminal_elision else 0)

    # ── declaration surfaces ───────────────────────────────────────
    def served_point(self) -> str:
        steps = ", ".join(f"{s}={n} steps" for s, n in sorted(self.steps.items()))
        guidance = ", ".join(f"{s}={m}@{w:g}" for s, (m, w) in sorted(self.guidance.items()))
        return f"{steps}; guidance {guidance}; cfg_batch={self.cfg_batch}"

    def key(self) -> str:
        gv = f"{self.video_guidance[0]}@{self.video_guidance[1]:g}"
        ga = f"{self.action_guidance[0]}@{self.action_guidance[1]:g}"
        return (f"v{self.video_steps}a{self.action_steps}_video-{gv}_action-{ga}"
                f"_b{self.cfg_batch}"
                + (f"_shift-v{self.video_shift:g}a{self.action_shift:g}"
                   if self.shifts != SHIFT else ""))

    def declaration(self) -> dict:
        """The ENGINE_BAKED_OPERATING_POINTS-shaped facts of this build (steps, guidance,
        evidence) — consumed by instinctflash.runtime.engine_backend.wan_va_baked_operating_point."""
        return {
            "backbone": "wan_va",
            "frontend": "serving/flash_rt/frontends/torch/wan_va_thor.py",
            "steps": dict(self.steps),
            "shifts": dict(self.shifts),
            "guidance": {s: tuple(g) for s, g in self.guidance.items()},
            "cfg_batch": self.cfg_batch,
            "forwards_per_cycle": self.forwards_per_cycle,
            "forwards_per_cycle_served_p010": self.forwards_per_cycle_served(True),
            "evidence": (
                f"the build sizes its AdaLN step tables from the declared grids (video t="
                f"{self.t_values('video')}, action t={self.t_values('action')}), allocates a "
                f"{self.cfg_batch}-stream KV slab, and runs {self.forwards_per_cycle} DiT forwards "
                f"per cycle; WanVaTorchFrontendThor.assert_point refuses any other request"),
        }

    def problem(self, requested_steps: Mapping, requested_guidance: Mapping, *,
                requester: str = "the request") -> Optional[str]:
        """ONE rule, the same one ``engine_operating_point_problem`` applies at plan/build time
        (serving/ cannot import instinctflash, so the rule is mirrored here and cross-checked
        by tests/test_wan_va_engine_operating_point.py). Returns the decline reason or None."""
        for stream in STREAMS:
            req = requested_steps.get(stream)
            try:
                req = int(req) if req is not None else None
            except (TypeError, ValueError):
                req = None
            n = self.steps[stream]
            if req is None:
                return (f"wan_va engine build bakes a {n}-step {stream} schedule, and the {stream} "
                        f"step count {requester} requests could not be verified — declining rather "
                        f"than running a schedule-baked pipeline blind")
            if req != n:
                return (f"wan_va engine build bakes a {n}-step {stream} schedule ({self.evidence_short()}); "
                        f"{requester} requests {stream}={req} — declining; the engine tier honors the "
                        f"operating point or declines, it never runs {n} steps under a plan that says {req}")
        for stream in STREAMS:
            req = requested_guidance.get(stream)
            mode, scale = self.guidance[stream]
            if req is None:
                return (f"wan_va engine build serves {stream} guidance {mode}@{scale:g}, and the "
                        f"{stream} guidance {requester} requests could not be verified — declining")
            want_neg, want_scale = _served_guidance(*req)
            have_neg, have_scale = _served_guidance(mode, scale)
            if want_neg != have_neg or (want_neg and want_scale != have_scale):
                want = f"{req[0]}@{'?' if req[1] is None else format(float(req[1]), 'g')}"
                return (f"wan_va engine build serves {stream} guidance {mode}@{scale:g} "
                        f"(cfg_batch={self.cfg_batch}); {requester} requests {stream}={want} — "
                        f"declining; a different guidance is a different computation and a "
                        f"different KV slab")
        return None

    def evidence_short(self) -> str:
        return (f"step tables video t={self.t_values('video')} / action t={self.t_values('action')}, "
                f"cfg_batch={self.cfg_batch}")

    def assert_serves(self, requested: "WanVaOperatingPoint", *, requester: str = "the request"):
        if self.shifts != requested.shifts:
            raise OperatingPointMismatch(
                f"wan_va engine schedule shifts {self.shifts} differ from "
                f"{requester}: {requested.shifts}")
        p = self.problem(requested.steps, requested.guidance, requester=requester)
        if p is not None:
            raise OperatingPointMismatch(p)

    # ── constructors from the worker surfaces ──────────────────────
    @classmethod
    def from_worker_flags(cls, degrade_nfe: str, guidance: Optional[str] = None,
                          *, default_video_scale: float = 5.0) -> "WanVaOperatingPoint":
        """The serve_variant CLI half: ``--degrade-nfe V,A`` and
        ``--guidance video=positive_only,action=positive_only`` (a mode name or a numeric scale).
        Without --guidance the family default applies (video cfg@5, action positive_only@1)."""
        v, a = (int(x) for x in str(degrade_nfe).split(","))
        g = {"video": ("cfg", float(default_video_scale)), "action": ("positive_only", 1.0)}
        for part in (guidance or "").split(","):
            if "=" not in part:
                continue
            k, val = (x.strip() for x in part.split("=", 1))
            if k not in STREAMS:
                raise ValueError(f"unknown guidance stream {k!r}")
            if val in MODES:
                g[k] = (val, 1.0) if val in _NO_NEGATIVE else ("cfg", float(default_video_scale))
            else:
                scale = float(val)
                g[k] = ("cfg", scale) if scale > 1.0 else ("positive_only", 1.0)
        return cls(v, a, g["video"], g["action"])

    @classmethod
    def from_job_config(cls, cfg) -> "WanVaOperatingPoint":
        """The stock server's resolved config (after serve_variant applied degrade-nfe and
        apply_declared_guidance): num_inference_steps / action_num_inference_steps /
        guidance_scale / action_guidance_scale. This is what the torch chain SERVES, so it is
        the request the engine must match."""
        gv = float(cfg.guidance_scale)
        ga = float(cfg.action_guidance_scale)
        return cls(int(cfg.num_inference_steps), int(cfg.action_num_inference_steps),
                   ("cfg", gv) if gv > 1 else ("positive_only", 1.0),
                   ("cfg", ga) if ga > 1 else ("positive_only", 1.0),
                   float(cfg.snr_shift), float(cfg.action_snr_shift))


POINT_2V4A_W5 = WanVaOperatingPoint(2, 4, ("cfg", 5.0), ("positive_only", 1.0))
POINT_2V2A_W1 = WanVaOperatingPoint(2, 2, ("positive_only", 1.0), ("positive_only", 1.0))
STAGE2_POINTS = {"2v4a_w5": POINT_2V4A_W5, "2v2a_w1": POINT_2V2A_W1}


def self_check() -> dict:
    p5, p1 = POINT_2V4A_W5, POINT_2V2A_W1
    assert p5.cfg_batch == 2 and p1.cfg_batch == 1
    assert p5.forwards_per_cycle == 10 and p1.forwards_per_cycle == 8
    assert p5.forwards_per_cycle_served() == 9 and p1.forwards_per_cycle_served() == 7
    assert p5.t_values("video") == [1000.0, 833.3333129882812, 0.0]
    assert p5.t_values("action") == [1000.0, 750.0, 500.0, 250.0, 0.0]
    assert p1.t_values("video") == [1000.0, 833.3333129882812, 0.0]
    assert p1.t_values("action") == [1000.0, 500.0, 0.0]
    assert p1.euler("action") == [-0.5, -0.5]
    assert p5.combine_scale("video") == 5.0 and p1.combine_scale("video") == 1.0
    assert p5.problem(p5.steps, p5.guidance) is None
    assert p5.problem(p1.steps, p1.guidance) is not None
    assert p1.problem(p5.steps, p5.guidance) is not None
    # cfg@1 == positive_only == none: same computation, accepted
    assert p1.problem(p1.steps, {"video": ("cfg", 1.0), "action": ("none", 1.0)}) is None
    # unverifiable stream declines
    assert p1.problem({"video": 2}, p1.guidance) is not None
    assert WanVaOperatingPoint.from_worker_flags("2,4") == p5
    assert WanVaOperatingPoint.from_worker_flags(
        "2,2", "video=positive_only,action=positive_only") == p1
    assert WanVaOperatingPoint.from_worker_flags("2,2", "video=1") == p1
    assert WanVaOperatingPoint.from_worker_flags("2,4", "video=5") == p5
    try:
        p5.assert_serves(p1)
        raise AssertionError("mismatch not raised")
    except OperatingPointMismatch:
        pass
    return {"2v4a_w5": p5.served_point(), "2v2a_w1": p1.served_point(),
            "keys": [p5.key(), p1.key()]}


__all__ = ["WanVaOperatingPoint", "OperatingPointMismatch", "POINT_2V4A_W5",
           "POINT_2V2A_W1", "STAGE2_POINTS", "STREAMS", "SHIFT", "MODES", "self_check"]
