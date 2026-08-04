"""LingBot-VA chunk-0 video stream, exposed as a PDD velocity oracle.

Everything backbone-specific about PDD-on-LingBot lives here, so that the `instinct-pdd` submodule
stays a general single-stream algorithm. Every fact below is cited to the source map in
`docs/PDD_ADAPTER_SOURCE_MAP.raw.json`, which is the canonical reference and should not be
regenerated.

WHY CHUNK 0 ONLY. Measured with `eval/lingbot_va_robotwin/probe_chunk0_cache.py`: after a reset the
KV cache holds 0 occupied slots, and it still holds 0 through every forward of the first chunk's
VIDEO denoise, because those run `update_cache=0` which writes slots and then restores them. So a
chunk-0 video training context carries no history term -- it is exactly (observation, prompt). The
same probe shows the ACTION stream reading the 7200 slots the video stream commits on its last step,
which is why action distillation is a separate stage and not a flag on this one.

THREE THINGS THIS FILE EXISTS TO GET RIGHT, each of which is silent when wrong:

  1. FRAME 0 IS NOT PART OF THE ODE STATE. `noisy_latents` is [1, 48, 2, 24, 20] and
     `_prepare_latent_input` overwrites frame 0 in place with the encoded observation and zeroes its
     timestep (wan_va_server.py:294-297). Frame 0 is clean conditioning. So the PDD state is frame 1
     alone, and this adapter assembles/strips the conditioning frame on every call. Handing PDD the
     full tensor would have it noise and integrate the observation.

  2. THE STUDENT MUST LEARN THE *GUIDED* VELOCITY. `forward_train` has no CFG path at all
     (modules/model.py:714) -- guidance exists only in the inference server. So the teacher target is
     built by running the real 2-row CFG batch and combining it exactly as serving does, while the
     student sees a single conditional context. That is what "guidance distilled in" means, and it is
     what removes the batch-2 duplication at serving time.

  3. THE GRID COMES FROM THE LIVE SCHEDULER, NOT FROM A RE-DERIVATION. The scheduler's own sigmas are
     handed straight to `Grid.from_times`. An earlier version re-derived the schedule and warped sigma
     in the wrong direction -- correct second grid point 991.7, re-derived 827.6 -- which clusters
     training steps at the data end while the sampler clusters them at the noise end.

  4. THE TIME AXIS IS FLIPPED AT THIS BOUNDARY. instinct-pdd integrates t ascending 0 -> 1 (the paper's
     convention); LingBot integrates sigma descending 1 -> 0. Since dt = -dsigma, a velocity is
     NEGATED on the way across, in `_Teacher.velocity` and `_Student.heads` and nowhere else. Both are
     negated, so the regression is unchanged -- but a single-sided flip would train against a target
     pointing backwards along the trajectory. See `grid()` for the full map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from instinct_pdd import Grid, MultiHeadStudent


@dataclass
class VideoContext:
    """One chunk-0 conditioning context: the clean observation frame, and nothing else.

    The prompt is not carried here because the server caches its T5 embeddings on itself at reset
    (`prompt_embeds` / `negative_prompt_embeds`, wan_va_server.py:424-433). Duplicating them would
    invite the two copies to disagree.
    """
    latent_cond: Any                  # [1, 48, 1, 24, 20], the encoded observation, bf16
    task: str = ""
    prompt: str = ""


class _CaptureProj(torch.nn.Module):
    """Wraps `transformer.proj_out` so one forward yields both the output and the trunk features.

    PDD's student is "the same backbone with the final linear layer repeated N times", so the trunk
    it needs is precisely the input to `proj_out`. Rather than reimplement the forward -- 30 blocks,
    AdaLN modulation, RoPE, the KV cache -- and risk it drifting from serving, this intercepts the
    real one. The captured tensor is byte-identical to what serving feeds `proj_out` by construction.

    Must be an `nn.Module`: `proj_out` is a registered child of the transformer, and
    `nn.Module.__setattr__` refuses to replace a child module with anything else.
    """

    def __init__(self, real):
        super().__init__()
        self.real = real
        self.captured = None

    def forward(self, x):
        self.captured = x
        return self.real(x)


class LingBotChunk0Video:
    """Factory for the teacher oracle, the student, and the grid, over a reset server.

    `server` must be a built, reset `VA_Server`: reset is what allocates the KV cache and encodes the
    prompt. Nothing here resets it, because a reset silently discards the conditioning context the
    caller just built.
    """

    #: RoboTwin geometry, from configs/va_robotwin_cfg.py and _reset's derivation.
    #: latent_height = ((256//16)*3)//2 = 24, latent_width = 320//16 = 20, z_dim = 48.
    LATENT_CHANNELS = 48

    def __init__(self, server, *, guidance: float = 5.0):
        self.S = server
        self.guidance = guidance
        # patch_size is (1, 2, 2) at every call site in the video path.
        self.patch_size = (1, 2, 2)

    # -- geometry ---------------------------------------------------------------------------------

    @property
    def frame_chunk(self) -> int:
        return int(self.S.job_config.frame_chunk_size)

    @property
    def n_denoise_frames(self) -> int:
        """Latent frames PDD actually integrates: all but the clean conditioning frame."""
        return self.frame_chunk - 1

    def state_shape(self) -> tuple:
        return (1, self.LATENT_CHANNELS, self.n_denoise_frames,
                int(self.S.latent_height), int(self.S.latent_width))

    def noise_like(self):
        """A tensor of the ODE state's shape, for the recipe's `<phase>/noise_like` slot."""
        import torch
        return torch.zeros(self.state_shape(), dtype=self.S.dtype, device=self.S.device)

    # -- the grid ---------------------------------------------------------------------------------

    def grid(self, n_intervals: int, block: int) -> Grid:
        """The exact sigma schedule an `n_intervals`-step video sampler walks, as an instinct-pdd Grid.

        THE CONVENTION BRIDGE, and it belongs here rather than in the algorithm. instinct-pdd fixes
        time ASCENDING from 0 (noise) to 1 (data), matching the paper's interpolant. LingBot-VA
        integrates sigma DESCENDING 1 -> 0. The map is t = 1 - sigma, in the same index order, so:

            times      = 1 - sigma            ascending, widths positive
            cond(i)    = sigma_i * 1000       which is (1 - t_i) * 1000
                       -> time_scale = -1000, time_offset = +1000

        and because dt = -dsigma, every velocity crossing this boundary flips sign. That is done in
        `_Teacher.velocity` and `_Student.heads`, once each, and nowhere else.

        Sigmas are taken off the live scheduler rather than recomputed. `set_timesteps` mutates it, so
        the server's own inference schedule is restored afterwards -- leaving it set to a training grid
        would silently change what a subsequent `_infer` does.
        """
        sch = self.S.scheduler
        prev = getattr(sch, "sigmas", None)
        prev_training = getattr(sch, "training", False)
        try:
            sch.set_timesteps(n_intervals)
            sigmas = [float(s) for s in sch.sigmas]
        finally:
            if prev is not None:
                sch.sigmas = prev
                sch.timesteps = prev * sch.num_train_timesteps
                sch.training = prev_training
        # The server pads a terminal sigma=0 (wan_va_server.py:473), so the grid ends on clean data.
        if abs(sigmas[-1]) > 1e-12:
            sigmas.append(0.0)
        scale = float(sch.num_train_timesteps)
        return Grid.from_times([1.0 - s for s in sigmas], block=block,
                               scale=-scale, offset=scale)

    # -- conditioning -----------------------------------------------------------------------------

    def encode_context(self, obs: dict, *, prompt: str, task: str = "") -> VideoContext:
        """Real observation -> real Wan VAE -> latent_cond, via the server's own encode path.

        `_encode_obs` is called rather than reimplemented: it owns the T-shape composition (wrists
        side by side along width, stacked above cam_high along height), the two separate VAE
        instances with independent feat_caches, the /255*2-1 normalisation and the latents_mean /
        latents_std standardisation. Reimplementing any of that would be a second source of truth.
        """
        import torch
        if prompt:
            # _reset(prompt) is what populates prompt_embeds and negative_prompt_embeds; without it
            # _prepare_latent_input faults on text_emb.
            self.S._reset(prompt=prompt)
        with torch.no_grad():
            init_latent = self.S._encode_obs(obs)
        # Serving takes only the first latent frame as the conditioning frame
        # (wan_va_server.py:491: init_latent[:, :, 0:1]).
        return VideoContext(latent_cond=init_latent[:, :, 0:1].to(self.S.dtype),
                            task=task, prompt=prompt)

    # -- the shared forward -----------------------------------------------------------------------

    #: Every forward runs two rows, always. Not a choice: the KV cache is allocated with
    #: batch_size=2 whenever `use_cfg` is on (wan_va_server.py:410), and RoboTwin's video guidance
    #: of 5 turns it on. A batch-1 forward writes batch-1 K/V into a batch-2 pool and comes back
    #: with batch 2 anyway -- which surfaced as a student head carrying 96 channels instead of 48,
    #: because the extra row got folded into the channel axis by the un-patchify.
    #:
    #: So the teacher runs [conditional, unconditional] and combines, while the student runs
    #: [conditional, conditional] and keeps row 0. The student's result is therefore numerically a
    #: single conditional forward, which is what "guidance distilled into the student" requires.
    #: Realising the 2x saving at serving time needs a batch-1 cache allocation; that is a runtime
    #: change, tracked separately, and it does not affect what the student learns here.
    CFG_BATCH = 2

    def _assemble(self, x, sigma_cond: float, ctx: VideoContext, batch: int, *, uncond: bool = True):
        """Build the transformer input dict for state `x` at conditioning time `sigma_cond`.

        The other three keys (timesteps with frame 0 zeroed, grid_id, text_emb) come from the
        server's own `_prepare_latent_input`; only `noisy_latents` is substituted, so the
        gradient-carrying tensor is ours while every convention stays theirs. `_prepare_latent_input`
        writes frame 0 IN PLACE, which is why it is handed a detached throwaway.
        """
        import torch
        full = torch.cat([ctx.latent_cond.to(x.dtype), x], dim=2)
        d = self.S._prepare_latent_input(
            full.detach().clone(), None, sigma_cond, sigma_cond,
            ctx.latent_cond, None, frame_st_id=0, patch_size=self.patch_size)["latent_res_lst"]
        d["noisy_latents"] = full                      # graph-carrying state, frame 0 already clean

        if uncond:
            # The real serving path: row 0 conditional, row 1 unconditional (empty-string T5).
            d = self.S._repeat_input_for_cfg(d)
        else:
            # Same tiling, but BOTH rows conditional -- so row 0 equals what a batch-1 conditional
            # forward would produce. Mirrors _repeat_input_for_cfg (wan_va_server.py:254-259)
            # exactly, except for the text_emb cat order.
            d["noisy_latents"] = d["noisy_latents"].repeat(batch, 1, 1, 1, 1)
            pos = self.S.prompt_embeds.to(self.S.dtype).clone()
            d["text_emb"] = torch.cat([pos] * batch, dim=0)
            d["grid_id"] = d["grid_id"][None].repeat(batch, 1, 1)
            d["timesteps"] = d["timesteps"][None].repeat(batch, 1)
        return d

    def _run(self, d, batch: int):
        """One transformer forward, un-patchified back to a latent video tensor.

        `update_cache=0` is what keeps this a chunk-0 call: slots are written and then restored, so
        the cache stays empty and consecutive calls do not condition on each other. Measured by
        probe_chunk0_cache.py.
        """
        from utils.utils import data_seq_to_patch
        out = self.S.transformer(d, update_cache=0, cache_name=self.S.cache_name,
                                 action_mode=False)
        return data_seq_to_patch(self.patch_size, out, self.frame_chunk,
                                 self.S.latent_height, self.S.latent_width, batch_size=batch)

    def _guided(self, pred):
        """CFG exactly as serving combines it (wan_va_server.py:513-516).

        `pred[1:] + s*(pred[:1] - pred[1:])` = uncond + s*(cond - uncond). Index 0 is conditional
        and index 1 unconditional, from the cat order in _repeat_input_for_cfg. Inverting these would
        train against a target pointing away from the prompt, and the loss would still fall.
        """
        return pred[1:] + self.guidance * (pred[:1] - pred[1:])

    # -- the two oracles --------------------------------------------------------------------------

    def teacher(self):
        return _Teacher(self)

    def student(self, n_heads: int):
        return _Student(self, n_heads)


class _Teacher:
    """`VelocityModel`: the frozen backbone, guided, returning only the denoisable frames."""

    def __init__(self, owner: LingBotChunk0Video):
        self.o = owner

    def velocity(self, x, t, *, cond: VideoContext = None):
        import torch
        if cond is None:
            raise ValueError("the chunk-0 video teacher needs a VideoContext; conditioning is not "
                             "optional for this stream")
        with torch.no_grad():
            d = self.o._assemble(x, float(t), cond, batch=self.o.CFG_BATCH, uncond=True)
            pred = self.o._run(d, batch=self.o.CFG_BATCH)
            guided = self.o._guided(pred)
        # Drop the clean conditioning frame, and flip sigma-velocity into t-velocity (dt = -dsigma).
        return -guided[:, :, 1:]


class _Student:
    """`MultiHeadVelocityModel`: N copies of `proj_out` over the shared trunk.

    Only the heads are trainable. The paper's architecture change is exactly "the final linear layer
    repeated N times, one per grid point", each initialised from the teacher's single final layer, so
    at step 0 every head reproduces the teacher's instantaneous velocity. Freezing the trunk on top
    of that is a deliberate first cut: it makes the overfit test cheap and isolates whether the
    objective is wired correctly from whether the backbone can fit it. Full fine-tuning is a later
    decision, not an oversight.

    The student runs a SINGLE conditional context -- no CFG batch -- because guidance is what it is
    being taught. See the module docstring.
    """

    def __init__(self, owner: LingBotChunk0Video, n_heads: int):
        import copy
        import torch

        self.o = owner
        self.n_heads = n_heads
        real = owner.S.transformer.proj_out
        self._capture = _CaptureProj(real)

        def make_head():
            h = copy.deepcopy(real)
            h = h.to(dtype=torch.float32, device=owner.S.device)
            # The backbone is frozen by `_configure_model`, which calls
            # `model.eval().requires_grad_(False)` -- and deepcopy inherits that, so a head copied
            # from proj_out arrives untrainable. Without this the loss computes fine and backward
            # fails with "element 0 of tensors does not require grad", several layers away from the
            # cause.
            return h.requires_grad_(True)

        self._mh = MultiHeadStudent(trunk_fn=self._trunk, make_head=make_head, n_heads=n_heads)
        self.head_list = self._mh.head_list

    def _trunk(self, x, t, cond):
        """Run the real backbone with `proj_out` intercepted, and return its input.

        The trunk is evaluated under no_grad: with only the heads trainable there is no gradient path
        into it, and building the graph would cost 30 blocks of activations for nothing.
        """
        import torch
        S = self.o.S
        prev = S.transformer.proj_out
        S.transformer.proj_out = self._capture
        try:
            with torch.no_grad():
                d = self.o._assemble(x, float(t), cond, batch=self.o.CFG_BATCH, uncond=False)
                S.transformer(d, update_cache=0, cache_name=S.cache_name, action_mode=False)
        finally:
            S.transformer.proj_out = prev
        feats = self._capture.captured
        if feats is None:
            raise RuntimeError(
                "proj_out was never called, so no trunk features were captured. The transformer's "
                "inference path changed shape; refusing to train a student on nothing.")
        return feats.detach().float()

    def heads(self, x, t, *, cond: VideoContext = None):
        """(n_heads, *state_shape) mean-velocity predictions from ONE backbone forward."""
        import torch
        from einops import rearrange

        from utils.utils import data_seq_to_patch

        if cond is None:
            raise ValueError("the chunk-0 video student needs a VideoContext")
        feats = self._trunk(x, t, cond)                      # [1, L, 3072], detached
        # Derive the un-patchify factor from the MODEL, not from an assumption. model.py:879-881
        # rearranges by prod(self.patch_size), and proj_out is Linear(inner, out_ch*prod(patch)) --
        # so the channel count that reaches data_seq_to_patch is out_features / prod(patch_size).
        # Asserting it here turns a silent channel mismatch deep inside `advance` into one message
        # carrying every number needed to see what disagreed.
        mp = tuple(getattr(self.o.S.transformer, "patch_size", self.o.patch_size))
        n = int(mp[0] * mp[1] * mp[2])
        out_features = int(self.head_list[0].out_features)
        c = out_features // n
        if c != self.o.LATENT_CHANNELS or out_features % n:
            raise RuntimeError(
                f"head geometry disagrees with the latent: proj_out.out_features={out_features}, "
                f"model.patch_size={mp} -> n={n}, so channels={out_features / n} but the latent has "
                f"{self.o.LATENT_CHANNELS}. Either patch_size is not what the un-patchify uses or "
                f"proj_out is not the video head.")
        out = []
        for h in self.head_list:
            y = h(feats)                                     # [1, L, out_ch*prod(patch)]
            y = rearrange(y, "b l (n c) -> b (l n) c", n=n)  # matches model.py:879-881
            y = data_seq_to_patch(self.o.patch_size, y, self.o.frame_chunk,
                                  self.o.S.latent_height, self.o.S.latent_width,
                                  batch_size=self.o.CFG_BATCH)
            # Row 0 is the conditional branch; rows are identical here by construction. Slicing
            # BEFORE returning is what keeps the student's output shape equal to the ODE state's.
            # Row 0 is the conditional branch (rows are identical here by construction). Negated for
            # the same reason as the teacher: this crosses into instinct-pdd's ascending-t convention.
            out.append(-y[:1, :, 1:].to(x.dtype))            # row 0, denoisable frames only
        return torch.stack(out, dim=0)

    def parameters(self, recurse: bool = True):
        return self._mh.parameters(recurse=recurse)

    def state_dict(self, *a, **k):
        return self._mh.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self._mh.load_state_dict(*a, **k)
