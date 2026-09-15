"""Engine-side routed-MoE path for the V2 action expert (T2-V2 M2 prep).

Implements the ``routed_moe_fn(layer, step, x_fp8, moe_out, stream)``
contract documented by ``pipeline_thor.routed_moe_stub`` on the new
flash_rt_kernels entries (csrc/kernels/moe_vla2.cu):

    moe_router_gemm_topk_fp8x / _fp16x / moe_router_topk
    fp8_gemm_batched_descale_fp16    (cuBLASLt strided-batched, v0.5)
    silu_mul_merged_fp8_fp16 / _fp16 (true SiLU — V2 experts)
    moe_combine_fp16 / _fp32

TWO ORTHOGONAL CHOICES, which ``mode`` used to conflate:

  * ``mode`` — the SCALE GRID the fp8 weights are quantized onto.
        "loop"    per-expert scales, shape (L, E) — the numerically-
                  checkable rung.
        "batched" ONE shared scale per (layer, matrix), shape (L,)
                  (the R3 decision: gateup spread <= 3.80x on all
                  layers, down <= 4.25x with only L13 over the ~4x
                  line — shared scales gated at M2d).
    Changing the grid CHANGES NUMERICS (R3 scale-mode delta: maxabs
    9.7–20.3 vs the per-expert grid, m2_kernel_unit_tests.json).

  * ``schedule`` — the LAUNCH SHAPE of the dense GEMMs.
        "loop"    v0: 64 batch-1 GEMM launches per layer.
        "batched" v0.5 ship shape: 2 strided-batched GEMMs per layer.
    At a fixed grid the schedules are BIT-IDENTICAL — same kernel,
    same fp8 bytes, same scales, only the launch shape differs
    (m2_kernel_unit_tests.json batched_gemm.gateup_broadcastA /
    _replicatedA: maxabs_batched_vs_batch1_loop = 0.0) — which is
    what makes the schedule an autotunable choice and the grid not.

``schedule`` defaults to following ``mode`` (yesterday's coupling,
bit-for-bit). ``schedule="auto"`` asks instinctflash's autotune to
measure both launch shapes on this device (cached per device+shape;
IFL_AUTOTUNE_VLA2_MOE_SCHEDULE forces one; the swap is verified
torch.equal before it is trusted); without instinctflash on the path
it falls back to the ship shape and says so. The "batched" schedule
requires the shared grid — one B-scale is all a strided-batched GEMM
can carry.

Router input: the same post-AdaRMS fp8 activation the experts read
(one tensor feeds router + routed + shared — repack_calib_plan §2).
The stock router sees the UNQUANTIZED bf16 hidden, so the fp8-input
router is a gated precision decision (M2d); ``router_source="fp16"``
uses a caller-provided fp16 buffer (e.g. the calibrate-path ``xn``)
via moe_router_gemm_topk_fp16x as the fallback.

Everything is shape-static and capture-safe. The cuBLASLt batched
entries build their descriptor cache on first call — run one eager
pass per shape before CUDA-graph capture (the vla4b/_capture_graphs
discipline already does this).

Torch parity baseline: moe_ref.py; unit tests:
/home/ubuntu/iwm_distill/thor_t2v2/m2_kernel_unit_tests.py.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

E, TOPK, MOE_H, HIDDEN = 32, 4, 512, 768
GU_OUT = 2 * MOE_H            # merged gate|up rows
ROUTED_SCALING = 4.0


def quantize_fp8_pt(w32: torch.Tensor, scale: Optional[float] = None):
    """Per-tensor symmetric E4M3 amax/448 (pi05/vla4b recipe).

    With an explicit ``scale`` this quantizes onto a caller-chosen grid
    (the shared-scale batched mode quantizes each expert's fp32 weight
    directly onto the per-(layer, matrix) shared grid — no
    double-rounding through the per-expert grid)."""
    if scale is None:
        scale = max(w32.abs().max().item() / 448.0, 1e-12)
    q = (w32 / scale).clamp(-448.0, 448.0).to(fp8)
    return q, float(scale)


class Vla2MoeEngine:
    """Holds MoE device weights + buffers; provides ``routed_moe_fn``.

    Weight layout per layer (NT convention — native HF [out, in]; the
    NN/[K,N] convention is cuBLASLt-unsupported on sm_90/cu12.8, see
    moe_vla2.cuh):
        gate_w   (E, 768) fp32   — router weight, fp32 end-to-end
        e_bias   (E,)     fp32   — e_score_correction_bias
        gu_fp8   (E, 1024, 768)  — merged gate|up, per expert [N, K]
        dn_fp8   (E, 768, 512)   — per expert [N, K]
        gu_scale / dn_scale      — fp32 device slots:
            mode="batched": shape (L,)   — shared per (layer, matrix)
            mode="loop":    shape (L, E) — per expert

    Activation scale slots (device fp32, pointer-stable):
        ffn-in  — EXTERNAL: the expert pipeline's act slot l*4+2 (the
                  post-AdaRMS quantize target; router + gateup + shared
                  all read it). Wire with ``bind_ffn_slots``.
        down-in — OWNED: (L,) engine slots for the routed-expert SiLU
                  intermediate (a site BEYOND the plan's 4/layer
                  budget; calibrated by m2_calibration.py or the
                  in-place calibrate pass).
    """

    def __init__(self, fvk_mod, num_layers: int = 36, *, tokens: int = 51,
                 mode: str = "batched", schedule: str | None = None,
                 router_source: str = "fp8"):
        assert mode in ("batched", "loop"), mode
        assert schedule in (None, "auto", "batched", "loop"), schedule
        assert router_source in ("fp8", "fp16"), router_source
        self.fvk = fvk_mod
        self.L = num_layers
        self.T = tokens
        self.mode = mode
        # None keeps yesterday's coupling: the grid's namesake launch shape. "auto" is
        # resolved by _finalize_schedule once weights and scales exist.
        self.schedule = schedule if schedule is not None else mode
        self.router_source = router_source

        self._gate_w: list[torch.Tensor] = []
        self._e_bias: list[torch.Tensor] = []
        self._gu_fp8: list[torch.Tensor] = []
        self._dn_fp8: list[torch.Tensor] = []
        self._gu_scale: Optional[torch.Tensor] = None   # device slots
        self._dn_scale: Optional[torch.Tensor] = None

        T = tokens
        self.gu_out = torch.empty(E, T, GU_OUT, dtype=fp16, device="cuda")
        self.h_fp8 = torch.zeros(E * T * MOE_H, dtype=torch.uint8,
                                 device="cuda")
        self.h_fp16 = torch.empty(E, T, MOE_H, dtype=fp16, device="cuda")
        self.slab = torch.empty(E, T, HIDDEN, dtype=fp16, device="cuda")
        self.ids = torch.zeros(T, TOPK, dtype=torch.int32, device="cuda")
        self.weights = torch.zeros(T, TOPK, dtype=torch.float32,
                                   device="cuda")
        self.logits = torch.zeros(T, E, dtype=torch.float32, device="cuda")
        # down-in act slots (owned; 1.0 until calibrated)
        self.dn_act = torch.full((num_layers,), 1.0, dtype=torch.float32,
                                 device="cuda")
        # ffn-in slots: external (pipeline act slot l*4+2); a private
        # fallback exists so the engine is testable standalone.
        self._own_ffn = torch.full((num_layers,), 1.0, dtype=torch.float32,
                                   device="cuda")
        self._ffn_base = self._own_ffn.data_ptr()
        self._ffn_stride = 4
        # fp16 router source buffer (router_source="fp16" only)
        self._x16_ptr = 0
        self.calibrate = False

    # ── weight loading ──────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, fvk_mod, checkpoint, *, tokens: int = 51,
                        mode: str = "batched", schedule: str | None = None,
                        router_source: str = "fp16"):
        """Quantize the supplied checkpoint directly, without old repack artifacts.

        This only constructs weights. The caller must bind activation/router
        buffers and calibrate the current deployment inputs before execution.
        """
        from .checkpoint_moe import checkpoint_moe_layers
        with checkpoint_moe_layers(checkpoint) as layers:
            return cls.from_fp32_layers(fvk_mod, layers, tokens=tokens, mode=mode,
                                       schedule=schedule, router_source=router_source)

    @classmethod
    def from_fp32_layers(cls, fvk_mod, layers, *, tokens: int = 51,
                         mode: str = "batched", schedule: str | None = None,
                         router_source: str = "fp8"):
        """``layers``: per-layer dicts of fp32 ckpt tensors
        (HF layout): gate.weight (32,768), e_score_correction_bias
        (32,), experts.gate_proj / up_proj (32,512,768),
        experts.down_proj (32,768,512).

        Quantizes straight from fp32 onto the mode's scale grid."""
        self = cls(fvk_mod, num_layers=len(layers), tokens=tokens,
                   mode=mode, schedule=schedule, router_source=router_source)
        gu_scales, dn_scales = [], []
        for lw in layers:
            gate_w = lw["gate.weight"].float().cuda().contiguous()
            e_bias = lw["e_score_correction_bias"].float().cuda().contiguous()
            # NT layout = native HF orientation, no transpose:
            # (E, 1024, 768) merged gate|up and (E, 768, 512) down
            gu32 = torch.cat([lw["experts.gate_proj"],
                              lw["experts.up_proj"]],
                             dim=1).float().cuda().contiguous()
            dn32 = lw["experts.down_proj"].float().cuda().contiguous()
            if mode == "batched":
                gq, gs = quantize_fp8_pt(gu32)   # ONE shared scale
                dq, ds = quantize_fp8_pt(dn32)
                gu_scales.append(gs); dn_scales.append(ds)
            else:
                gq = torch.empty_like(gu32, dtype=fp8)
                dq = torch.empty_like(dn32, dtype=fp8)
                gss, dss = [], []
                for e in range(E):
                    q, s = quantize_fp8_pt(gu32[e]); gq[e] = q; gss.append(s)
                    q, s = quantize_fp8_pt(dn32[e]); dq[e] = q; dss.append(s)
                gu_scales.append(gss); dn_scales.append(dss)
            self._gate_w.append(gate_w)
            self._e_bias.append(e_bias)
            self._gu_fp8.append(gq)
            self._dn_fp8.append(dq)
        self._gu_scale = torch.tensor(gu_scales, dtype=torch.float32,
                                      device="cuda")
        self._dn_scale = torch.tensor(dn_scales, dtype=torch.float32,
                                      device="cuda")
        self._finalize_schedule()
        return self

    @classmethod
    def from_frontend(cls, frontend, *, mode: str = "batched",
                      schedule: str | None = None,
                      router_source: str = "fp8"):
        """Wire from a constructed Vla2TorchFrontendThor (its ``_moe``
        dict holds per-expert-scale fp8). mode="loop" reuses those
        bytes; mode="batched" REQUANTIZES onto the shared grid (one
        extra fp8 rounding vs quantizing from fp32 — repack_v2.py
        artifacts carry the direct-from-fp32 shared variant instead).
        Binds the pipeline's ffn-in act slots."""
        import flash_rt.flash_rt_kernels as fvk_mod
        moe = frontend._moe
        L = len(moe["gate_w32"])
        self = cls(fvk_mod, num_layers=L, mode=mode, schedule=schedule,
                   router_source=router_source)
        for l in range(L):
            self._gate_w.append(moe["gate_w32"][l].contiguous())
            self._e_bias.append(moe["e_bias32"][l].contiguous())
        # frontend stores [K, N]-transposed fp8 (the NN convention) —
        # flip to the NT orientation used here
        def to_nt(q):
            return q.transpose(-1, -2).contiguous()

        if mode == "loop":
            for l in range(L):
                self._gu_fp8.append(to_nt(moe["gateup_fp8"][l]))
                self._dn_fp8.append(to_nt(moe["down_fp8"][l]))
            self._gu_scale = torch.tensor(moe["gateup_scales"],
                                          dtype=torch.float32, device="cuda")
            self._dn_scale = torch.tensor(moe["down_scales"],
                                          dtype=torch.float32, device="cuda")
        else:
            gu_s, dn_s = [], []
            for l in range(L):
                gs = max(moe["gateup_scales"][l])
                ds = max(moe["down_scales"][l])
                pe_g = torch.tensor(moe["gateup_scales"][l], device="cuda"
                                    ).view(E, 1, 1)
                pe_d = torch.tensor(moe["down_scales"][l], device="cuda"
                                    ).view(E, 1, 1)
                self._gu_fp8.append(to_nt(quantize_fp8_pt(
                    moe["gateup_fp8"][l].float() * pe_g, gs)[0]))
                self._dn_fp8.append(to_nt(quantize_fp8_pt(
                    moe["down_fp8"][l].float() * pe_d, ds)[0]))
                gu_s.append(gs); dn_s.append(ds)
            self._gu_scale = torch.tensor(gu_s, dtype=torch.float32,
                                          device="cuda")
            self._dn_scale = torch.tensor(dn_s, dtype=torch.float32,
                                          device="cuda")
        self.bind_ffn_slots(frontend._exp_act_scales.data_ptr() + 2 * 4, 16)
        self._finalize_schedule()
        return self

    @classmethod
    def from_artifacts(cls, fvk_mod, moe_art: dict, *, tokens: int = 51,
                       mode: str = "batched", schedule: str | None = None,
                       router_source: str = "fp8"):
        """Load from a repack_v2.py ``moe.pt`` dict (both scale modes
        are materialized in the artifact; picks the requested one)."""
        L = moe_art["num_layers"]
        self = cls(fvk_mod, num_layers=L, tokens=tokens, mode=mode,
                   schedule=schedule, router_source=router_source)
        key = "shared" if mode == "batched" else "per_expert"
        for l in range(L):
            self._gate_w.append(moe_art["gate_w"][l].float().cuda())
            self._e_bias.append(moe_art["e_bias"][l].float().cuda())
            self._gu_fp8.append(moe_art[f"gateup_fp8_{key}"][l].cuda())
            self._dn_fp8.append(moe_art[f"down_fp8_{key}"][l].cuda())
        self._gu_scale = moe_art[f"gateup_scale_{key}"].float().cuda()
        self._dn_scale = moe_art[f"down_scale_{key}"].float().cuda()
        self._finalize_schedule()
        return self

    # ── launch schedule ─────────────────────────────────────────

    def _finalize_schedule(self):
        """Resolve schedule="auto" and enforce grid legality, once weights and scales exist.

        The "batched" schedule needs the shared grid: a strided-batched GEMM carries ONE
        B-scale, so on the per-expert grid it would descale 31 of 32 experts wrongly --
        refused here rather than served."""
        if self.schedule == "auto":
            self.schedule = resolve_moe_schedule(self)
        if self.schedule == "batched" and self._gu_scale is not None \
                and self._gu_scale.dim() != 1:
            raise ValueError(
                "schedule='batched' on the per-expert scale grid (mode='loop'): a "
                "strided-batched GEMM carries one B-scale per launch, so this pairing would "
                "be numerically wrong. Use schedule='loop', or quantize onto the shared grid "
                "(mode='batched').")

    def _gu_scale_ptr(self, l: int, e: int = 0) -> int:
        """Per-expert weight-scale slot for the LOOP schedule; grid-agnostic on purpose --
        on the shared grid every expert of a layer reads the same slot, which is exactly
        what makes the loop/batched swap bit-identical there."""
        if self._gu_scale.dim() == 1:
            return self._gu_scale.data_ptr() + l * 4
        return self._gu_scale.data_ptr() + (l * E + e) * 4

    def _dn_scale_ptr(self, l: int, e: int = 0) -> int:
        if self._dn_scale.dim() == 1:
            return self._dn_scale.data_ptr() + l * 4
        return self._dn_scale.data_ptr() + (l * E + e) * 4

    # ── act-scale plumbing ──────────────────────────────────────

    def bind_ffn_slots(self, base_ptr: int, stride_bytes: int):
        """ffn-in slot for layer l lives at base_ptr + l*stride_bytes
        (pipeline layout: exp_act_scales.data_ptr() + (l*4+2)*4 →
        base = data_ptr()+8, stride 16)."""
        self._ffn_base = base_ptr
        self._ffn_stride = stride_bytes

    def bind_router_fp16(self, x16_ptr: int):
        """fp16 router-source buffer (router_source='fp16' mode)."""
        self._x16_ptr = x16_ptr

    def set_down_act_scales(self, amax_per_layer):
        """Write calibrated down-in scales (amax/448) into the owned
        slots — pointer-stable, no recapture."""
        vals = torch.as_tensor(amax_per_layer, dtype=torch.float32) / 448.0
        assert vals.numel() == self.L
        self.dn_act.copy_(vals.cuda())

    def _ffn_slot(self, l: int) -> int:
        return self._ffn_base + l * self._ffn_stride

    def _dn_slot(self, l: int) -> int:
        return self.dn_act.data_ptr() + l * 4

    # ── the routed_moe_fn contract ──────────────────────────────

    def routed_moe_fn(self, l: int, s: int, x_fp8: int, moe_out: int,
                      stream: int):
        """pipeline_thor.expert_forward hook: x_fp8 = post-AdaRMS fp8
        (quantized with the ffn-in slot), moe_out = [T, 768] fp16
        routed contribution (shared expert NOT included)."""
        fvk = self.fvk
        T = self.T

        # 1. router (fp32 math end-to-end)
        if self.router_source == "fp8":
            fvk.moe_router_gemm_topk_fp8x(
                x_fp8, self._gate_w[l].data_ptr(),
                self._e_bias[l].data_ptr(), self.ids.data_ptr(),
                self.weights.data_ptr(), self.logits.data_ptr(),
                T, HIDDEN, E, TOPK, ROUTED_SCALING,
                self._ffn_slot(l), stream)
        else:
            assert self._x16_ptr, "bind_router_fp16 first"
            fvk.moe_router_gemm_topk_fp16x(
                self._x16_ptr, self._gate_w[l].data_ptr(),
                self._e_bias[l].data_ptr(), self.ids.data_ptr(),
                self.weights.data_ptr(), self.logits.data_ptr(),
                T, HIDDEN, E, TOPK, ROUTED_SCALING, stream)

        # 2. gate|up GEMM(s), dense over all 32 experts (NT layout). The branch is the LAUNCH
        # SHAPE only; both read the same fp8 bytes and, on the shared grid, the same scales --
        # the M2-proven bit-identical pair (maxabs_batched_vs_batch1_loop = 0.0).
        if self.schedule == "batched":
            fvk.fp8_gemm_batched_descale_nt_fp16(
                x_fp8, self._gu_fp8[l].data_ptr(), self.gu_out.data_ptr(),
                T, GU_OUT, HIDDEN, E,
                0, GU_OUT * HIDDEN, T * GU_OUT,          # strideA=0: broadcast
                self._ffn_slot(l), self._gu_scale.data_ptr() + l * 4, stream)
        else:
            for e in range(E):
                fvk.fp8_gemm_batched_descale_nt_fp16(
                    x_fp8, self._gu_fp8[l][e].data_ptr(),
                    self.gu_out.data_ptr() + e * T * GU_OUT * 2,
                    T, GU_OUT, HIDDEN, 1, 0, 0, 0,
                    self._ffn_slot(l),
                    self._gu_scale_ptr(l, e), stream)

        # 3. true-SiLU gate*up → fp8 on the down-in slot
        if self.calibrate:
            fvk.silu_mul_merged_fp16(self.gu_out.data_ptr(),
                                     self.h_fp16.data_ptr(),
                                     E * T, MOE_H, stream)
            fvk.quantize_fp8_device_fp16(self.h_fp16.data_ptr(),
                                         self.h_fp8.data_ptr(),
                                         self._dn_slot(l),
                                         E * T * MOE_H, stream)
        else:
            fvk.silu_mul_merged_fp8_fp16(self.gu_out.data_ptr(),
                                         self.h_fp8.data_ptr(),
                                         E * T, MOE_H,
                                         self._dn_slot(l), stream)

        # 4. down GEMM(s) → dense slab (E, T, 768) (NT layout); same launch-shape-only branch
        if self.schedule == "batched":
            fvk.fp8_gemm_batched_descale_nt_fp16(
                self.h_fp8.data_ptr(), self._dn_fp8[l].data_ptr(),
                self.slab.data_ptr(),
                T, HIDDEN, MOE_H, E,
                T * MOE_H, HIDDEN * MOE_H, T * HIDDEN,
                self._dn_slot(l), self._dn_scale.data_ptr() + l * 4, stream)
        else:
            for e in range(E):
                fvk.fp8_gemm_batched_descale_nt_fp16(
                    self.h_fp8.data_ptr() + e * T * MOE_H,
                    self._dn_fp8[l][e].data_ptr(),
                    self.slab.data_ptr() + e * T * HIDDEN * 2,
                    T, HIDDEN, MOE_H, 1, 0, 0, 0,
                    self._dn_slot(l),
                    self._dn_scale_ptr(l, e), stream)

        # 5. weighted top-4 combine (routed only — pipeline adds shared)
        fvk.moe_combine_fp16(self.slab.data_ptr(), self.ids.data_ptr(),
                             self.weights.data_ptr(), moe_out, 0,
                             T, HIDDEN, TOPK, stream)


# ── schedule autotune (instinctflash site) ──────────────────────

MOE_SCHEDULE_SITE_NAME = "vla2_moe_schedule"


def moe_schedule_site(engine: "Vla2MoeEngine"):
    """The autotune Site for this engine's launch schedule. Imported lazily so flash_rt stays
    deployable without instinctflash; raises ImportError where it is absent."""
    from instinctflash.autotune import Candidate, Site, register_site
    from instinctflash.passes.contract import Tier

    grid = "shared" if (engine._gu_scale is not None and engine._gu_scale.dim() == 1) \
        else "per_expert"
    return register_site(Site(
        name=MOE_SCHEDULE_SITE_NAME,
        candidates=(
            Candidate(
                "batched", Tier.BITEXACT,
                evidence=("the v0.5 ship shape: 2 strided-batched cuBLASLt GEMMs + fused "
                          "SiLU + combine per layer (R3/M2d)")),
            Candidate(
                "loop", Tier.BITEXACT,
                evidence=(
                    "M2 unit tests (iwm_distill/thor_t2v2/m2_kernel_unit_tests.json), cases "
                    "batched_gemm.gateup_broadcastA and _replicatedA: "
                    "maxabs_batched_vs_batch1_loop = 0.0 -- same NT kernel, same fp8 bytes, "
                    "same scales, only the launch shape differs. NOT the R3 scale-mode delta "
                    "(per-expert vs shared grids, maxabs 9.7-20.3): the grid is a separate, "
                    "numerics-changing decision and is not part of this site. The swap is "
                    "additionally verified torch.equal at tune time, on this device")),
        ),
        baseline="batched",
        shape_signature=(f"E{E} T{engine.T} moe_h{MOE_H} h{HIDDEN} "
                         f"L{engine.L} grid={grid}"),
    ))


def resolve_moe_schedule(engine: "Vla2MoeEngine", *,
                         model_id: str = "lingbot-vla-v2") -> str:
    """Resolve schedule='auto': measure both launch shapes on this device, verified, cached.

    Falls back to the grid's legal default -- loudly -- when instinctflash is not importable:
    a standalone flash_rt deploy must keep working, and the ship shape is the measured default
    it always had. On the per-expert grid there is nothing to tune ('batched' is illegal
    there), so 'loop' is returned without a bench.
    """
    if engine._gu_scale is not None and engine._gu_scale.dim() != 1:
        logger.info("schedule='auto': per-expert scale grid, where the batched schedule is "
                    "illegal (one B-scale per launch); using 'loop' -- nothing to tune.")
        return "loop"
    try:
        from instinctflash.autotune import autotune
    except ImportError:
        logger.warning(
            "schedule='auto': instinctflash is not importable here, so the schedule cannot "
            "be autotuned; keeping the v0.5 ship shape ('batched'). Install instinctflash "
            "or pass schedule= explicitly to silence this.")
        return "batched"

    site = moe_schedule_site(engine)
    T = engine.T
    x16 = torch.randn(T, HIDDEN, device="cuda", dtype=fp16)
    xq, xs = quantize_fp8_pt(x16.float())
    x_fp8 = xq.cuda().contiguous()
    out = torch.zeros(T, HIDDEN, dtype=fp16, device="cuda")
    prev = engine.schedule

    def _run(schedule: str) -> None:
        engine.schedule = schedule
        for l in range(engine.L):
            engine.routed_moe_fn(l, 0, x_fp8.data_ptr(), out.data_ptr(), 0)

    def bench(cand) -> float:
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        _run(cand.name)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    def verify(cand_name: str) -> float:
        """max|delta| between the candidate's output and the baseline's, same input. The site
        claims BITEXACT, so anything but 0.0 keeps the baseline (audit_tier's stance)."""
        _run(site.baseline)
        torch.cuda.synchronize()
        base = out.clone()
        _run(cand_name)
        torch.cuda.synchronize()
        return 0.0 if torch.equal(out, base) else (out.float() - base.float()).abs().max().item()

    try:
        decision = autotune(site, bench, model_id=model_id, verify=verify)
    finally:
        engine.schedule = prev
    logger.info("%s", decision.reason)
    engine.autotune_decision = decision
    return decision.chosen


__all__ = ["Vla2MoeEngine", "moe_schedule_site", "quantize_fp8_pt", "resolve_moe_schedule"]
