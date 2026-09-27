"""GI LingBot-VLA-V2 engine: Qwen3-VL text prefix + 36-layer MoE flow-matching expert.

Prefix rows (compact; padded language tokens are dropped, which upstream masks anyway):

    3 x [vision_start, 64 patches, vision_end] | L language tokens | 16 align tokens

The prefix is causal with interleaved M-RoPE positions (computed per prompt on the
host); deepstack features are added to the patch rows after layers 0-2. The suffix
[state, 50 actions] attends the valid prefix (state: + itself; actions: + everything).
"""
from __future__ import annotations

import numpy as np
import torch

from instinctflash.engines.common import Arena, GraphCache, bucket
from instinctflash.kernels.activation import SILU, gated_act_quant, moe_act_quant
from instinctflash.kernels.attention import attention
from instinctflash.kernels.fp8 import E4M3, gemm
from instinctflash.kernels.norm import residual_norm_quant, row_quant
from instinctflash.kernels.rope import qkv_split
from instinctflash.engines.vla2.weights import EXPERT_FF, EXPERTS, ROUTE_SCALE, SHARED_FF, TOP_K

CAMS, PATCHES = 3, 64
BLOCK = PATCHES + 2
VISION_ROWS = CAMS * BLOCK          # 198
ALIGN = 16
D, FF, EX_D = 2560, 9728, 768
HQ, HKV, HD = 32, 8, 128
QKV_N = (HQ + 2 * HKV) * HD
MOE_F = EXPERTS * EXPERT_FF + SHARED_FF
STATE_DIM = 55

ATTENTION_SITES = {
    "prefix": (64, 64, 4, 2, True),
    "expert": (32, 64, 4, 2, False),
}


class Vla2Engine:
    def __init__(self, weights, device, *, max_lang=72, chunk=50, bucket_size=16):
        self.w = weights
        self.dev = torch.device(device)
        self.chunk, self.suffix = int(chunk), int(chunk) + 1
        self.max_lang = int(max_lang)
        self.bucket_size = int(bucket_size)
        self.attn = {k: (v[:4], v[4]) for k, v in ATTENTION_SITES.items()}
        dev, f32, f16, bf16 = self.dev, torch.float32, torch.float16, torch.bfloat16
        tail = bucket(self.max_lang + ALIGN, self.bucket_size)
        smax = VISION_ROWS + tail
        self.smax = smax
        self.vis = torch.zeros(CAMS * PATCHES, D, dtype=f32, device=dev)
        self.deep = torch.zeros(3, CAMS * PATCHES, D, dtype=f32, device=dev)
        self.tail = torch.zeros(tail, D, dtype=f32, device=dev)
        self.state = torch.zeros(1, STATE_DIM, dtype=f32, device=dev)
        self.noise = torch.zeros(self.chunk, STATE_DIM, dtype=f32, device=dev)
        self.prefix_len = torch.zeros(1, dtype=torch.int32, device=dev)
        rows = max(smax, self.suffix)
        self.q8 = Arena(rows * max(FF, MOE_F), E4M3, dev)
        self.raw = Arena(rows * 2 * max(FF, MOE_F), bf16, dev)
        self.scale = [torch.zeros(rows, dtype=f32, device=dev) for _ in range(2)]
        self.px = torch.zeros(smax, D, dtype=f32, device=dev)
        self.q16 = torch.zeros(smax, HQ, HD, dtype=f16, device=dev)
        self.o16 = torch.zeros(smax, HQ * HD, dtype=f16, device=dev)
        self.kc = torch.zeros(len(weights.lm), smax + self.suffix, HKV, HD, dtype=f16, device=dev)
        self.vc = torch.zeros_like(self.kc)
        self.x_t = torch.zeros(self.chunk, STATE_DIM, dtype=f32, device=dev)
        self.dx = torch.zeros(self.suffix, EX_D, dtype=f32, device=dev)
        self.dy = torch.zeros(self.suffix, EX_D, dtype=f32, device=dev)
        self.hid = torch.zeros(self.chunk, EX_D, dtype=f32, device=dev)
        self.state_emb = torch.zeros(1, EX_D, dtype=f32, device=dev)
        self.route = torch.zeros(self.suffix, EXPERTS, dtype=f32, device=dev)
        self.pos_prefix = torch.zeros(smax, 3, dtype=f32, device=dev)
        self.pos_suffix = torch.zeros(self.suffix, 3, dtype=f32, device=dev)
        self.kv_extra = torch.full((self.suffix,), self.suffix, dtype=torch.int32, device=dev)
        self.kv_extra[0] = 1
        self.act_w_t = weights.act_w.t().contiguous()
        self.mlp_out_w_t = weights.mlp_out_w.t().contiguous()
        self.state_w_t = weights.state_w.t().contiguous()
        self.out_w_dt = (weights.out_w * weights.dt).t().contiguous()
        self.out_b_dt = (weights.out_b * weights.dt).contiguous()
        self._graphs = GraphCache(dev)
        self._rows = None
        self._host_tail = torch.zeros(tail, D, dtype=f32).pin_memory()
        self._host_pos = torch.zeros(smax + self.suffix, 3, dtype=f32).pin_memory()
        self.graph_captured = False
        self.prompt_updates = 0
        self._fill_vision_positions()

    def _fill_vision_positions(self):
        pos = self._host_pos
        r = torch.arange(PATCHES) // 8
        c = torch.arange(PATCHES) % 8
        for k in range(CAMS):
            base, row = 10 * k, BLOCK * k
            pos[row] = base
            pos[row + 1: row + 1 + PATCHES, 0] = base + 1
            pos[row + 1: row + 1 + PATCHES, 1] = base + 1 + r
            pos[row + 1: row + 1 + PATCHES, 2] = base + 1 + c
            pos[row + 1 + PATCHES] = base + 9

    # ------------------------------------------------------------------ inputs
    def set_prompt(self, token_ids):
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64))
        n = int(ids.numel())
        if not 1 <= n <= self.max_lang:
            raise ValueError(f"VLA2 engine supports 1..{self.max_lang} prompt tokens, got {n}")
        tail = self._host_tail
        tail.zero_()
        tail[:n] = self.w.embed.index_select(0, ids)
        tail[n:n + ALIGN] = self.w.align.cpu()
        pos = self._host_pos
        pos[VISION_ROWS:] = 0
        seq = 30 + torch.arange(n + ALIGN, dtype=torch.float32)
        pos[VISION_ROWS:VISION_ROWS + n + ALIGN] = seq[:, None]
        p0 = 30 + n + ALIGN
        pos[self.smax:self.smax + self.suffix] = (p0 + torch.arange(self.suffix, dtype=torch.float32))[:, None]
        self.tail.copy_(tail, non_blocking=True)
        self.pos_prefix.copy_(pos[:self.smax], non_blocking=True)
        self.pos_suffix.copy_(pos[self.smax:], non_blocking=True)
        self.prefix_len.fill_(VISION_ROWS + n + ALIGN)
        self._rows = VISION_ROWS + bucket(n + ALIGN, self.bucket_size)
        self.prompt_updates += 1

    def infer(self, merged, deepstack, state, noise):
        """merged / deepstack[i]: [192, 2560]; state [1, 55]; noise [50, 55]. Returns x_t [50, 55]."""
        if self._rows is None:
            raise RuntimeError("set_prompt must be called before infer")
        self.vis.copy_(merged.reshape(CAMS * PATCHES, D))
        for i in range(3):
            self.deep[i].copy_(deepstack[i].reshape(CAMS * PATCHES, D))
        self.state.copy_(state.reshape(1, STATE_DIM))
        self.noise.copy_(noise.reshape(self.chunk, STATE_DIM))
        self._graphs.run(self._rows, self._forward)
        self.graph_captured = True
        return self.x_t

    # ----------------------------------------------------------------- forward
    def _forward(self, M):
        self._prefix(M)
        self._expert()

    def _patch_rows(self, t):
        return t.view(-1, D)[:VISION_ROWS].view(CAMS, BLOCK, D)[:, 1:1 + PATCHES]

    def _prefix(self, M):
        w, q8, raw, s = self.w, self.q8, self.raw, self.scale
        blocks = self.px[:VISION_ROWS].view(CAMS, BLOCK, D)
        blocks[:, 0] = w.vision_start
        blocks[:, 1 + PATCHES] = w.vision_end
        blocks[:, 1:1 + PATCHES] = self.vis.view(CAMS, PATCHES, D)
        self.px[VISION_ROWS:M].copy_(self.tail[: M - VISION_ROWS])
        residual_norm_quant(self.px[:M], q=q8(M, D), q_scale=s[0])
        qv = self.q16[:M].view(1, M, HQ, HD)
        cfg, tc5 = self.attn["prefix"]
        for i, L in enumerate(w.lm):
            gemm(q8(M, D), L.qkv, raw(M, QKV_N))
            qkv_split(raw(M, QKV_N), self.q16[:M], self.kc[i], self.vc[i], heads_q=HQ, heads_kv=HKV,
                      head_dim=HD, row=s[0], col=L.qkv.scale, inv_freq=w.inv_freq, pos=self.pos_prefix,
                      sections=w.sections, q_norm=L.q_norm, k_norm=L.k_norm)
            if L.o is None:
                break
            attention(qv, self.kc[i][None], self.vc[i][None], out=self.o16[:M].view(1, M, HQ, HD),
                      kv_len=self.prefix_len, causal=True, config=cfg, tc5=tc5)
            row_quant(self.o16[:M], q8(M, HQ * HD), s[1], num_warps=1)
            gemm(q8(M, HQ * HD), L.o, raw(M, D))
            residual_norm_quant(self.px[:M], delta=raw(M, D), delta_row=s[1], delta_col=L.o.scale,
                                q=q8(M, D), q_scale=s[0], num_warps=16)
            gemm(q8(M, D), L.gate_up, raw(M, 2 * FF))
            gated_act_quant(raw(M, 2 * FF), q8(M, FF), s[1], row=s[0], col=L.gate_up.scale, act=SILU, num_warps=8)
            gemm(q8(M, FF), L.down, raw(M, D))
            if i < 3:
                # Deepstack: add this layer's visual features to the patch rows before the next norm.
                residual_norm_quant(self.px[:M], delta=raw(M, D), delta_row=s[1], delta_col=L.down.scale)
                self._patch_rows(self.px).add_(self.deep[i].view(CAMS, PATCHES, D))
                residual_norm_quant(self.px[:M], q=q8(M, D), q_scale=s[0], num_warps=16)
            else:
                residual_norm_quant(self.px[:M], delta=raw(M, D), delta_row=s[1], delta_col=L.down.scale,
                                    q=q8(M, D), q_scale=s[0], num_warps=16)

    def _expert(self):
        w, q8, raw, s = self.w, self.q8, self.raw, self.scale
        S = self.suffix
        torch.addmm(w.state_b, self.state, self.state_w_t, out=self.state_emb)
        self.x_t.copy_(self.noise)
        qv = self.q16[:S].view(1, S, HQ, HD)
        ov = self.o16[:S].view(1, S, HQ, HD)
        cfg, tc5 = self.attn["expert"]
        last = len(w.expert) - 1
        for step in range(len(w.times)):
            ma, mf = w.mod_attn[step], w.mod_ffn[step]
            self.dx[:1].copy_(self.state_emb)
            torch.addmm(w.act_c[step], self.x_t, self.act_w_t, out=self.hid)
            torch.nn.functional.silu(self.hid, inplace=True)
            torch.addmm(w.mlp_out_b, self.hid, self.mlp_out_w_t, out=self.dx[1:])
            residual_norm_quant(self.dx, mod_scale=ma[0, 0], mod_shift=ma[0, 1], q=q8(S, EX_D), q_scale=s[0])
            for i, L in enumerate(w.expert):
                gemm(q8(S, EX_D), L.qkv, raw(S, QKV_N))
                qkv_split(raw(S, QKV_N), self.q16[:S], self.kc[i], self.vc[i], heads_q=HQ, heads_kv=HKV,
                          head_dim=HD, row=s[0], col=L.qkv.scale, bias=L.qkv.bias, inv_freq=w.inv_freq,
                          pos=self.pos_suffix, sections=w.sections, kv_base=self.prefix_len)
                attention(qv, self.kc[i][None], self.vc[i][None], out=ov, kv_len=self.prefix_len,
                          kv_extra=self.kv_extra, max_extra=S, config=cfg, tc5=tc5)
                row_quant(self.o16[:S], q8(S, HQ * HD), s[1], num_warps=1)
                gemm(q8(S, HQ * HD), L.o, raw(S, EX_D))
                residual_norm_quant(self.dx, delta=raw(S, EX_D), delta_row=s[1], delta_col=L.o.scale,
                                    mod_scale=mf[i, 0], mod_shift=mf[i, 1], q=q8(S, EX_D), q_scale=s[0],
                                    router=(L.router, L.router_bias, self.route, TOP_K, ROUTE_SCALE))
                gemm(q8(S, EX_D), L.gate_up, raw(S, 2 * MOE_F))
                moe_act_quant(raw(S, 2 * MOE_F), q8(S, MOE_F), s[1], self.route, row=s[0], col=L.gate_up.scale,
                              experts=EXPERTS, group=EXPERT_FF, n_routed=EXPERTS * EXPERT_FF)
                gemm(q8(S, MOE_F), L.down, raw(S, EX_D))
                if i < last:
                    residual_norm_quant(self.dx, delta=raw(S, EX_D), delta_row=s[1], delta_col=L.down.scale,
                                        mod_scale=ma[i + 1, 0], mod_shift=ma[i + 1, 1], q=q8(S, EX_D), q_scale=s[0])
                else:
                    residual_norm_quant(self.dx, delta=raw(S, EX_D), delta_row=s[1], delta_col=L.down.scale,
                                        weight=w.final_norm, y=self.dy)
            self.x_t.addmm_(self.dy[1:], self.out_w_dt).add_(self.out_b_dt)
