"""Checkpoint geometry for the VA DiT, after the native VAE camera layout."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WanVaGeometry:
    frame_chunk: int = 2
    latent_h: int = 24
    latent_w: int = 20
    action_per_frame: int = 16

    def __post_init__(self):
        for name in ("frame_chunk", "latent_h", "latent_w", "action_per_frame"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.latent_h % 2 or self.latent_w % 2:
            raise ValueError("VA latent dimensions must be divisible by the 2x2 patch")

    @property
    def tokens_per_frame(self):
        return (self.latent_h // 2) * (self.latent_w // 2)

    @property
    def video_tokens(self):
        return self.frame_chunk * self.tokens_per_frame

    @property
    def action_tokens(self):
        return self.frame_chunk * self.action_per_frame

    @property
    def max_video_tokens(self):
        return (self.frame_chunk + 1) * self.tokens_per_frame

    @property
    def max_tokens(self):
        return max(self.max_video_tokens, self.action_tokens)

    @classmethod
    def from_job_config(cls, cfg):
        if tuple(cfg.patch_size) != (1, 2, 2):
            raise ValueError("VA engine requires checkpoint patch_size=(1, 2, 2)")
        if cfg.action_dim != 30:
            raise ValueError("VA engine requires 30 normalized action channels")
        if cfg.env_type == "robotwin_tshape":
            h, w = ((cfg.height // 16) * 3) // 2, cfg.width // 16
        elif cfg.env_type == "none":
            h, w = cfg.height // 16, cfg.width // 16 * len(cfg.obs_cam_keys)
        else:
            raise ValueError(f"Unsupported VA camera layout: {cfg.env_type!r}")
        return cls(cfg.frame_chunk_size, h, w, cfg.action_per_frame)

    def pool_slots(self, attn_window):
        # This divisor is two even for F=4: it is the native allocator formula.
        if type(attn_window) is not int or attn_window < 2:
            raise ValueError("attn_window must be an integer >= 2")
        return (attn_window // 2) * (self.video_tokens + self.action_tokens)
