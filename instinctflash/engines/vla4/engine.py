"""GI LingBot-VLA-4B engine: Qwen2.5 text prefix + 36-layer flow-matching expert.

    vision embeddings [192, 2048] (native BF16 tower) + token embeddings -> prefix rows
    prefix (36 layers, bidirectional, 1D RoPE theta 1e4) -> per-layer K/V
    suffix [state, 50 actions] -> 10 Euler steps of the expert; each expert layer attends
    its LM layer's K/V plus the suffix (state row: prefix + itself; actions: everything)

One CUDA graph per prompt-length bucket; the valid prompt length, rotary positions and
suffix placement live in device memory. Per-token FP8 activations, per-channel weights.
"""
from __future__ import annotations

import numpy as np
import torch

from instinctflash.engines.common import Arena, GraphCache, bucket
from instinctflash.kernels.activation import SILU, gated_act_quant
from instinctflash.kernels.attention import attention
from instinctflash.kernels.fp8 import E4M3, gemm
from instinctflash.kernels.norm import residual_norm_quant, row_quant
from instinctflash.kernels.rope import qkv_split, rope_inv_freq

VISION_TOKENS = 192
D, FF, EX_D, EX_FF = 2048, 11008, 768, 2752
HQ, HKV, HD = 16, 2, 128
QKV_N = (HQ + 2 * HKV) * HD
STATE_DIM = 75

ATTENTION_SITES = {
    "prefix": (64, 64, 4, 2, True),
    "expert": (32, 64, 4, 2, False),
}


class Vla4Engine:
    def __init__(self, weights, device, *, max_lang=72, chunk=50, bucket_size=16):
        self.w = weights
        self.dev = torch.device(device)
        self.chunk = int(chunk)
        self.suffix = self.chunk + 1
        self.max_lang = int(max_lang)
        self.bucket_size = int(bucket_size)
        self.attn = {k: (v[:4], v[4]) for k, v in ATTENTION_SITES.items()}
        dev, f32, f16, bf16 = self.dev, torch.float32, torch.float16, torch.bfloat16
        smax = VISION_TOKENS + bucket(self.max_lang, self.bucket_size)
        self.smax = smax
        self.vis = torch.zeros(VISION_TOKENS, D, dtype=f32, device=dev)
        self.lang = torch.zeros(bucket(self.max_lang, self.bucket_size), D, dtype=f32, device=dev)
        self.state = torch.zeros(1, STATE_DIM, dtype=f32, device=dev)
        self.noise = torch.zeros(self.chunk, STATE_DIM, dtype=f32, device=dev)
        self.prefix_len = torch.zeros(1, dtype=torch.int32, device=dev)
        rows = max(smax, self.suffix)
        self.q8 = Arena(rows * FF, E4M3, dev)
        self.raw = Arena(rows * 2 * FF, bf16, dev)
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
        self.inv_freq = rope_inv_freq(HD, 10000.0, dev)
        self.pos_prefix = torch.arange(smax, dtype=f32, device=dev)[:, None].contiguous()
        self.pos_suffix = torch.zeros(self.suffix, 1, dtype=f32, device=dev)
        # Suffix visibility beyond the valid prefix: state row sees itself, actions see all 51.
        self.kv_extra = torch.full((self.suffix,), self.suffix, dtype=torch.int32, device=dev)
        self.kv_extra[0] = 1
        self.act_w_t = weights.act_w.t().contiguous()
        self.mlp_out_w_t = weights.mlp_out_w.t().contiguous()
        self.state_w_t = weights.state_w.t().contiguous()
        self.out_w_dt = (weights.out_w * weights.dt).t().contiguous()
        self.out_b_dt = (weights.out_b * weights.dt).contiguous()
        self._graphs = GraphCache(dev)
        self._rows = None
        self._host_lang = torch.zeros(self.max_lang, D, dtype=f32).pin_memory()
        self.graph_captured = False
        self.prompt_updates = 0

    # ------------------------------------------------------------------ inputs
    def set_prompt(self, token_ids):
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64))
        n = int(ids.numel())
        if not 1 <= n <= self.max_lang:
            raise ValueError(f"VLA4 engine supports 1..{self.max_lang} prompt tokens, got {n}")
        self._host_lang[:n].copy_(self.w.embed.index_select(0, ids))
        self.lang[n:].zero_()
        self.lang[:n].copy_(self._host_lang[:n], non_blocking=True)
        prefix = VISION_TOKENS + n
        self.prefix_len.fill_(prefix)
        torch.arange(prefix, prefix + self.suffix, out=self.pos_suffix[:, 0])
        self._rows = VISION_TOKENS + bucket(n, self.bucket_size)
        self.prompt_updates += 1

    def infer(self, vision, state, noise):
        """vision [192, 2048]; state [1, 75] normalized; noise [50, 75]. Returns x_t [50, 75]."""
        if self._rows is None:
            raise RuntimeError("set_prompt must be called before infer")
        self.vis.copy_(vision.reshape(VISION_TOKENS, D))
        self.state.copy_(state.reshape(1, STATE_DIM))
        self.noise.copy_(noise.reshape(self.chunk, STATE_DIM))
        self._graphs.run(self._rows, self._forward)
        self.graph_captured = True
        return self.x_t

    # ----------------------------------------------------------------- forward
    def _forward(self, M):
        self._prefix(M)
        self._expert()

    def _prefix(self, M):
        w, q8, raw, s = self.w, self.q8, self.raw, self.scale
        V = VISION_TOKENS
        self.px[:V].copy_(self.vis)
        self.px[V:M].copy_(self.lang[: M - V])
        residual_norm_quant(self.px[:M], q=q8(M, D), q_scale=s[0])
        qv = self.q16[:M].view(1, M, HQ, HD)
        cfg, tc5 = self.attn["prefix"]
        for i, L in enumerate(w.lm):
            gemm(q8(M, D), L.qkv, raw(M, QKV_N))
            qkv_split(raw(M, QKV_N), self.q16[:M], self.kc[i], self.vc[i], heads_q=HQ, heads_kv=HKV,
                      head_dim=HD, row=s[0], col=L.qkv.scale, bias=L.qkv.bias, inv_freq=self.inv_freq,
                      pos=self.pos_prefix)
            if L.o is None:
                break
            attention(qv, self.kc[i][None], self.vc[i][None], out=self.o16[:M].view(1, M, HQ, HD),
                      kv_len=self.prefix_len, config=cfg, tc5=tc5)
            row_quant(self.o16[:M], q8(M, HQ * HD), s[1], num_warps=1)
            gemm(q8(M, HQ * HD), L.o, raw(M, D))
            residual_norm_quant(self.px[:M], delta=raw(M, D), delta_row=s[1], delta_col=L.o.scale,
                                q=q8(M, D), q_scale=s[0], num_warps=16)
            gemm(q8(M, D), L.gate_up, raw(M, 2 * FF))
            gated_act_quant(raw(M, 2 * FF), q8(M, FF), s[1], row=s[0], col=L.gate_up.scale, act=SILU, num_warps=8)
            gemm(q8(M, FF), L.down, raw(M, D))
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
                          head_dim=HD, row=s[0], col=L.qkv.scale, bias=L.qkv.bias, inv_freq=self.inv_freq,
                          pos=self.pos_suffix, kv_base=self.prefix_len)
                attention(qv, self.kc[i][None], self.vc[i][None], out=ov, kv_len=self.prefix_len,
                          kv_extra=self.kv_extra, max_extra=S, config=cfg, tc5=tc5)
                row_quant(self.o16[:S], q8(S, HQ * HD), s[1], num_warps=1)
                gemm(q8(S, HQ * HD), L.o, raw(S, EX_D))
                residual_norm_quant(self.dx, delta=raw(S, EX_D), delta_row=s[1], delta_col=L.o.scale,
                                    mod_scale=mf[i, 0], mod_shift=mf[i, 1], q=q8(S, EX_D), q_scale=s[0])
                gemm(q8(S, EX_D), L.gate_up, raw(S, 2 * EX_FF))
                gated_act_quant(raw(S, 2 * EX_FF), q8(S, EX_FF), s[1], row=s[0], col=L.gate_up.scale, act=SILU)
                gemm(q8(S, EX_FF), L.down, raw(S, EX_D))
                if i < last:
                    residual_norm_quant(self.dx, delta=raw(S, EX_D), delta_row=s[1], delta_col=L.down.scale,
                                        mod_scale=ma[i + 1, 0], mod_shift=ma[i + 1, 1], q=q8(S, EX_D), q_scale=s[0])
                else:
                    residual_norm_quant(self.dx, delta=raw(S, EX_D), delta_row=s[1], delta_col=L.down.scale,
                                        weight=w.final_norm, y=self.dy)
            self.x_t.addmm_(self.dy[1:], self.out_w_dt).add_(self.out_b_dt)
