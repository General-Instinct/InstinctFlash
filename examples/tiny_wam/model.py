"""A tiny but REAL world-action model. Small enough to publish, shaped like the real thing.

This is not a toy in the sense of "fake". It is a real `torch.nn.Module` with real weights that runs
a real forward pass and produces a real action chunk. It is a toy in the sense of "1.2 MB instead of
24 GB", which is what makes it publishable as a worked example.

WHY IT IS SHAPED THE WAY IT IS. Every field the checkpoint declares has to mean something, or the
example proves nothing:

  nfe                  the model really does run N denoise forwards per control step
  guidance             the video stream really does duplicate the batch for CFG; the action stream
                       really does not
  output_projection    the action head really is L linear heads over an N-interval grid, and they
                       really are foldable into one affine map at load time

So `capabilities()` on the published checkpoint describes this module accurately, and a pass admitted
by those capabilities would be admitted correctly.

WHAT IT IS NOT. It is not trained -- the weights are seeded random. It predicts nothing useful. No
accuracy claim is made or possible. The claim is about the WORKFLOW: declare, publish, resolve, plan,
run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN = 64
LAYERS = 2
HEADS = 4
OBS_DIM = 32
ACTION_DIM = 14          # a 7-DoF bimanual arm, as a plausible shape
ACTION_HORIZON = 8
N_INTERVALS = 8
BLOCK = 4                # -> nfe = n_intervals // block = 2 action forwards


class Block(nn.Module):
    def __init__(self, hidden: int = HIDDEN, heads: int = HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Linear(hidden * 2, hidden))

    def forward(self, x, temb):
        h = self.norm1(x + temb)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class TinyWAM(nn.Module):
    """Observation -> latent -> N denoise forwards -> action chunk.

    The action head is `n_intervals` linear maps. At a given denoise step the model uses a *block* of
    `block` consecutive heads, and because each is linear their weighted sum is itself a single
    affine map -- which is the whole content of `output_projection.foldable`. `fold_heads()` performs
    that fold, and `forward(..., folded=True)` uses the result. The two paths agree to floating-point
    round-off, which `run_end_to_end.py` checks rather than asserts.
    """

    def __init__(self):
        super().__init__()
        self.obs_in = nn.Linear(OBS_DIM, HIDDEN)
        self.t_embed = nn.Sequential(nn.Linear(1, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm_out = nn.LayerNorm(HIDDEN)
        # n_intervals linear velocity heads, one per interval of the noise schedule
        self.heads = nn.ModuleList([nn.Linear(HIDDEN, ACTION_DIM) for _ in range(N_INTERVALS)])

    # -- the capability that `output_projection.foldable` declares -----------------------------
    def fold_heads(self, interval_block: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold one block of linear heads into a single affine map (W, b).

        L linear heads averaged over a block is one linear map. Folding at load time means the
        student costs the same per forward as a single-head model, instead of paying L head
        evaluations for a result that is a fixed linear combination.
        """
        lo = interval_block * BLOCK
        sel = list(self.heads)[lo:lo + BLOCK]
        W = torch.stack([h.weight for h in sel]).mean(0)
        b = torch.stack([h.bias for h in sel]).mean(0)
        return W, b

    def forward(self, obs: torch.Tensor, *, nfe: int = 2, cfg_scale: float = 0.0,
                folded: bool = True) -> torch.Tensor:
        """One control step. Returns an action chunk (B, ACTION_HORIZON, ACTION_DIM)."""
        b = obs.shape[0]
        x = self.obs_in(obs).unsqueeze(1).expand(b, ACTION_HORIZON, HIDDEN).contiguous()

        for step in range(nfe):
            t = torch.full((b, 1), 1.0 - step / max(nfe, 1), device=obs.device, dtype=obs.dtype)
            temb = self.t_embed(t).unsqueeze(1)

            h = x
            if cfg_scale > 0:                       # the video stream's CFG: a duplicated batch
                h = torch.cat([h, h], dim=0)
                temb = torch.cat([temb, temb], dim=0)
            for blk in self.blocks:
                h = blk(h, temb)
            if cfg_scale > 0:
                uncond, cond = h.chunk(2, dim=0)
                h = uncond + cfg_scale * (cond - uncond)

            h = self.norm_out(h)
            if folded:
                W, bias = self.fold_heads(step % (N_INTERVALS // BLOCK))
                v = torch.nn.functional.linear(h, W, bias)
            else:
                lo = (step % (N_INTERVALS // BLOCK)) * BLOCK
                v = torch.stack([hd(h) for hd in list(self.heads)[lo:lo + BLOCK]]).mean(0)

            # sigma-descending velocity: x moves along v by the interval width
            x = x + (1.0 / nfe) * torch.nn.functional.pad(v, (0, HIDDEN - ACTION_DIM))

        W, bias = self.fold_heads(0)
        return torch.nn.functional.linear(self.norm_out(x), W, bias)


def build(seed: int = 0) -> TinyWAM:
    torch.manual_seed(seed)
    m = TinyWAM()
    m.eval()
    return m
