"""AdaLN step tables for LingBot-VA (wan_va) at the fixed 2V/4A schedule.

Ground truth (`wan_va/modules/model.py` + CPU-verified dtype map, 2026-08-28):

* Per block: ``scale_shift_table`` [1,6,3072] + ``timestep_proj`` -> 6 vectors
  (shift/scale/gate for attn, then for ffn) (model:512-513, :524-532);
  modulation ``LN_no_affine(x)*(1+scale)+shift`` in fp32, gated residual in
  fp32, all cast back to bf16 (:534-536, :543-544, :558-565). Head: root
  ``scale_shift_table`` [1,2,3072] + **temb** (not proj) (:646-651, :867-874).
* DEPLOYED dtype chain (diffusers ``_keep_in_fp32_modules`` measured on the
  real ckpt loaded with torch_dtype=bf16): sinusoid fp32 (``Timesteps`` 256ch,
  flip_sin_to_cos=True, shift 0) -> ``time_embedder`` MLP **fp32** ->
  ``.to(bf16)`` (model:238) -> SiLU **bf16** -> ``time_proj`` GEMM **bf16**
  -> ``.float()`` + **fp32** ``scale_shift_table`` (values are bf16 — the
  ckpt stores BF16 throughout, so keeping-in-fp32 loses nothing; only the
  ARITHMETIC chain matters).
* Timesteps are per-frame repeat-interleaved (model:851-858): one t per
  forward everywhere EXCEPT the episode-first infer/commit, where frame 0 is
  pinned to t=0 -> two contiguous row spans (frame-major token order) -> two
  span-wise ``ada_layer_norm_fp16`` calls with two table rows.
* The VIDEO stream uses ``condition_embedder``; the ACTION stream uses
  ``condition_embedder_action`` (model:855-856). NOTE the `_action` copy's
  text_embedder is DEAD at inference (model:843 always uses the video one).
* Fixed schedules (CPU-verified from ``FlowMatchScheduler``, fp32-exact):
  video (NFE 2, shift 5.0): t = [1000.0, 833.3333129882812, 0.0], the last
  forward NEVER steps the scheduler (server:508) — it exists to commit
  pred-KV at t=0; sigma-deltas [-0.16666669, -0.83333331].
  action (NFE 4, shift 1.0): t = [1000, 750, 500, 250, 0], deltas [-0.25]*4.
  kv-commit forwards run at t=0 for all tokens of the committed stream.
* Euler STATE accumulates in bf16 (scheduler.step keeps sample dtype —
  verified); the engine loop must re-round latents/actions to bf16 per step.

M0 gate: byte-compare these rows against the stock
``_time_embed``/``scale_shift_table`` chain on the SAME device (bf16 GEMM
reduction order is device-dependent; build tables on the gate device when
byte parity is demanded).

Pure PyTorch; diffusers-free fallback implements the identical math and is
cross-checked against the diffusers modules in ``self_check()`` when
diffusers is importable.
"""
from __future__ import annotations

import math

import torch

D = 3072
FREQ_DIM = 256
STEPS_VIDEO, STEPS_ACTION = 2, 4
SHIFT_VIDEO, SHIFT_ACTION = 5.0, 1.0
NUM_TRAIN_TIMESTEPS = 1000


# ────────────────────────────────────────────────────────────────────
# Schedule (FlowMatchScheduler.set_timesteps, extra_one_step, sigma_min=0)
# ────────────────────────────────────────────────────────────────────

def flow_match_sigmas(n: int, shift: float) -> torch.Tensor:
    """fp32 sigmas, byte-for-byte scheduler.py:33-58 (training=False)."""
    s = torch.linspace(1.0, 0.0, n + 1)[:-1]
    return shift * s / (1 + (shift - 1) * s)


def forward_t_values(n: int, shift: float) -> list[float]:
    """The padded timestep list the server iterates (server:468-483):
    n scheduler timesteps + one trailing 0.0 (the pred-KV commit forward)."""
    t = flow_match_sigmas(n, shift) * NUM_TRAIN_TIMESTEPS
    return [float(v) for v in t] + [0.0]


def euler_deltas(n: int, shift: float) -> list[float]:
    """(sigma_next - sigma) per SCHEDULER step (the last forward never steps
    — server:508/:548). The final step's sigma_next is 0 (sched:83-85)."""
    s = flow_match_sigmas(n, shift)
    out = []
    for i in range(n):
        nxt = 0.0 if i + 1 >= n else float(s[i + 1])
        out.append(nxt - float(s[i]))
    return out


# ────────────────────────────────────────────────────────────────────
# The deployed time-embedding chain
# ────────────────────────────────────────────────────────────────────

def sinusoid_fp32(t: float, dim: int = FREQ_DIM) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True,
    downscale_freq_shift=0)`` in fp32: half=dim/2, exponent
    -ln(10000)*arange(half)/half, output cat[cos, sin]."""
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    emb = torch.tensor(float(t), dtype=torch.float32) * exponent.exp()
    return torch.cat([emb.cos(), emb.sin()])


class EmbedderWeights:
    """One condition_embedder's time chain, fp32 hosts (bf16 ckpt values).

    Keys (under ``condition_embedder{,_action}.``):
        time_embedder.linear_1.{weight,bias}  [3072,256]/[3072]
        time_embedder.linear_2.{weight,bias}  [3072,3072]/[3072]
        time_proj.{weight,bias}               [18432,3072]/[18432]
    """

    def __init__(self, te1_w, te1_b, te2_w, te2_b, tp_w, tp_b):
        self.te1_w = te1_w.float()
        self.te1_b = te1_b.float()
        self.te2_w = te2_w.float()
        self.te2_b = te2_b.float()
        # time_proj is a bf16 module at deploy — keep bf16 hosts
        self.tp_w = tp_w.to(torch.bfloat16)
        self.tp_b = tp_b.to(torch.bfloat16)

    def temb_and_proj(self, t: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (temb bf16 [3072], proj fp32 [6,3072]) — the deployed
        chain: fp32 sinusoid -> fp32 MLP (SiLU inside) -> bf16 cast ->
        bf16 SiLU -> bf16 GEMM -> fp32."""
        F_ = torch.nn.functional
        ts = sinusoid_fp32(t)
        # F.linear IS the op the stock modules run — correct-by-
        # construction (a hand-rolled mv differs by isolated ulps on CPU
        # at K=3072; byte parity vs the STOCK module on the gate device
        # is the M0 requirement, so use the identical op here)
        h = F_.silu(F_.linear(ts, self.te1_w, self.te1_b))     # fp32 module
        temb = F_.linear(h, self.te2_w, self.te2_b).to(torch.bfloat16)
        act = F_.silu(temb)                                    # bf16
        proj = F_.linear(act, self.tp_w, self.tp_b)            # bf16 GEMM
        return temb, proj.float().view(6, D)


def build_stream_tables(
    emb: EmbedderWeights,
    block_tables: list,          # 30 x [1,6,3072] (or [6,3072]) fp32 hosts
    root_table: torch.Tensor,    # [1,2,3072] (or [2,3072]) fp32 host
    t_values: list,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Per-stream step tables.

    Returns dict:
        mod   (T, L, 6, 3072) dtype — shift_msa, scale_msa, gate_msa,
                                      c_shift, c_scale, c_gate (chunk order
                                      of model:524-532)
        out   (T, 2, 3072)   dtype — norm_out shift, scale (from temb,
                                      model:867-874)
        t_values, and the kv-commit row is the t=0 entry (present in both
        streams' schedules by construction).
    """
    L = len(block_tables)
    T = len(t_values)
    mod = torch.empty(T, L, 6, D, dtype=torch.float32)
    out = torch.empty(T, 2, D, dtype=torch.float32)
    for ti, t in enumerate(t_values):
        temb, proj = emb.temb_and_proj(float(t))
        for l in range(L):
            mod[ti, l] = block_tables[l].float().view(6, D) + proj
        out[ti] = root_table.float().view(2, D) + temb.float()[None, :]
    return {"mod": mod.to(dtype), "out": out.to(dtype),
            "t_values": [float(t) for t in t_values]}


# ────────────────────────────────────────────────────────────────────
# Self-check: fallback chain vs the real diffusers modules (bitwise)
# ────────────────────────────────────────────────────────────────────

def self_check() -> dict:
    torch.manual_seed(1)
    report = {}

    tv = forward_t_values(STEPS_VIDEO, SHIFT_VIDEO)
    ta = forward_t_values(STEPS_ACTION, SHIFT_ACTION)
    assert tv == [1000.0, 833.3333129882812, 0.0], tv
    assert ta == [1000.0, 750.0, 500.0, 250.0, 0.0], ta
    dv, da = euler_deltas(2, SHIFT_VIDEO), euler_deltas(4, SHIFT_ACTION)
    assert abs(dv[1] + 0.8333333134651184) < 1e-9 and len(dv) == 2
    assert da == [-0.25, -0.25, -0.25, -0.25]
    report["t_video"], report["t_action"] = tv, ta
    report["euler_video"], report["euler_action"] = dv, da
    # Stage-2 second point (2V/2A@w1, h2_thor_realtime_design §5.1): the action grid is
    # {1000, 500} + the t=0 commit row, Euler deltas [-0.5, -0.5]; video grid unchanged.
    ta2 = forward_t_values(2, SHIFT_ACTION)
    assert ta2 == [1000.0, 500.0, 0.0], ta2
    assert euler_deltas(2, SHIFT_ACTION) == [-0.5, -0.5]
    report["t_action_2step"] = ta2
    # the 1-step action grid (the H2 student's) for completeness: {1000} + commit row
    assert forward_t_values(1, SHIFT_ACTION) == [1000.0, 0.0]
    assert euler_deltas(1, SHIFT_ACTION) == [-1.0]

    # random-weight embedder, fallback chain vs diffusers modules
    te1_w = torch.randn(D, FREQ_DIM).to(torch.bfloat16).float()
    te1_b = torch.randn(D).to(torch.bfloat16).float()
    te2_w = (torch.randn(D, D) / 55).to(torch.bfloat16).float()
    te2_b = torch.randn(D).to(torch.bfloat16).float()
    tp_w = (torch.randn(6 * D, D) / 55).to(torch.bfloat16).float()
    tp_b = torch.randn(6 * D).to(torch.bfloat16).float()
    emb = EmbedderWeights(te1_w, te1_b, te2_w, te2_b, tp_w, tp_b)

    try:
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    except Exception:
        report["diffusers_crosscheck"] = "SKIPPED (diffusers unavailable)"
        return report

    tsp = Timesteps(num_channels=FREQ_DIM, flip_sin_to_cos=True,
                    downscale_freq_shift=0)
    te = TimestepEmbedding(in_channels=FREQ_DIM, time_embed_dim=D)  # fp32
    with torch.no_grad():
        te.linear_1.weight.copy_(te1_w); te.linear_1.bias.copy_(te1_b)
        te.linear_2.weight.copy_(te2_w); te.linear_2.bias.copy_(te2_b)
    proj_mod = torch.nn.Linear(D, 6 * D).to(torch.bfloat16)
    with torch.no_grad():
        proj_mod.weight.copy_(tp_w.to(torch.bfloat16))
        proj_mod.bias.copy_(tp_b.to(torch.bfloat16))

    worst_temb = worst_proj = 0.0
    for t in tv + ta + ta2:
        # stock chain (model:226-240 with the measured dtype map)
        ts = tsp(torch.tensor([float(t)]))                # fp32 sinusoid
        temb_ref = te(ts)[0].to(torch.bfloat16)           # fp32 MLP -> bf16
        proj_ref = proj_mod(torch.nn.functional.silu(temb_ref)).float()
        temb, proj = emb.temb_and_proj(float(t))
        worst_temb = max(worst_temb,
                         (temb.float() - temb_ref.float()).abs().max().item())
        worst_proj = max(worst_proj,
                         (proj.reshape(-1) - proj_ref).abs().max().item())
    # fp32 parts must be bitwise; the bf16 GEMM may differ in reduction
    # order between mv and linear — assert bitwise here (CPU both paths are
    # a plain mv); if a backend changes this, the M0 gate compares against
    # the STOCK module on the gate device instead.
    assert worst_temb == 0.0, worst_temb
    assert worst_proj == 0.0, worst_proj
    report["diffusers_crosscheck_temb_maxdiff"] = worst_temb
    report["diffusers_crosscheck_proj_maxdiff"] = worst_proj

    # table assembly shape/route check
    blocks = [torch.randn(1, 6, D).to(torch.bfloat16).float() for _ in range(3)]
    root = torch.randn(1, 2, D).to(torch.bfloat16).float()
    tb = build_stream_tables(emb, blocks, root, tv)
    assert tb["mod"].shape == (3, 3, 6, D) and tb["out"].shape == (3, 2, D)
    report["table_shapes_ok"] = True
    return report


__all__ = [
    "D", "FREQ_DIM", "STEPS_VIDEO", "STEPS_ACTION",
    "SHIFT_VIDEO", "SHIFT_ACTION",
    "flow_match_sigmas", "forward_t_values", "euler_deltas",
    "sinusoid_fp32", "EmbedderWeights", "build_stream_tables", "self_check",
]
