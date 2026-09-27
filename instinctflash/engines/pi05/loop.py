"""Runtime loop for the GI pi05 engine.

Keeps the checkpoint's own LeRobot processors (tokenization with state,
image resize/normalization, action unnormalization) and replaces only the
action generator. Images stay on the device end to end; one engine (and its
graphs) is kept per active-camera count, all sharing one prepared weight set.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch

from instinctflash.runtime.pi05_engine import Pi05EngineLoop


class GiPi05Loop(Pi05EngineLoop):
    def __init__(self, config, pre, post, preprocess_images, weights, device):
        super().__init__(config, pre, post, preprocess_images, device)
        self._weights = weights
        self.backend_stats = {"backend": "gi_pi05_fp8", "precision": "fp8",
                              "activation_scales": "per-token dynamic", "weight_scales": "per-channel"}

    def _engine(self, views):
        from instinctflash.engines.pi05.engine import Pi05Engine
        eng = self._frontends.get(views)
        if eng is None:
            eng = self._frontends[views] = Pi05Engine(self._weights, views, self._device,
                                                      chunk=int(self.config.chunk_size))
        return eng

    def _refill(self, batch):
        torch = self._torch
        images, masks = self._preprocess_images(batch)
        active = []
        for image, mask in zip(images, masks):
            if image.shape[0] != 1 or mask.numel() != 1:
                raise ValueError("pi05 Runtime currently serves one observation at a time")
            if bool(mask.item()):
                active.append(image[0])
        if not 1 <= len(active) <= 3:
            raise ValueError("pi05 GI engine requires one to three active cameras")
        tokens = batch["observation.language.tokens"]
        mask = batch["observation.language.attention_mask"].bool()
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape != mask.shape:
            raise ValueError("pi05 token IDs and masks must have matching [1, length] shapes")
        ids = tokens[mask].detach().cpu().numpy()
        eng = self._engine(len(active))
        with torch.cuda.device(self._device) if self._device.type == "cuda" else nullcontext():
            eng.set_prompt(ids)
            raw = eng.infer(torch.stack(active))
        if tuple(raw.shape) != (int(self.config.chunk_size), 32):
            raise RuntimeError(f"pi05 GI engine returned unexpected action shape {tuple(raw.shape)}")
        actions = raw[:, :self._action_dim].clone()
        if not torch.isfinite(actions).all():
            raise RuntimeError("pi05 GI engine returned nonfinite actions")
        self._queue.extend(actions[:self.config.n_action_steps].unsqueeze(1))

    def close(self):
        super().close()
        self._weights = None


def build_pi05_engine_loop(checkpoint, *, device=None, action_steps=None):
    from pathlib import Path
    from types import SimpleNamespace
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from instinctflash.engines.pi05.weights import load_pi05_weights

    repo = str(Path(checkpoint.path))
    dev = torch.device(device or "cuda")
    if dev.type != "cuda":
        raise ValueError("pi05 GI engine requires a CUDA device")
    cfg = PI05Config.from_pretrained(repo)
    cfg.device = str(dev)
    cfg.compile_model = False
    cfg.validate_features()
    if action_steps is not None:
        cfg.num_inference_steps = int(action_steps)
    if (cfg.max_action_dim != 32 or cfg.paligemma_variant != "gemma_2b"
            or cfg.action_expert_variant != "gemma_300m"):
        raise ValueError("pi05 GI engine requires gemma_2b/gemma_300m with max_action_dim=32")
    if getattr(cfg, "rtc_config", None) is not None:
        raise ValueError("pi05 GI engine action queue does not yet support RTC")
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=repo,
        preprocessor_overrides={"device_processor": {"device": str(dev)}},
        postprocessor_overrides={"device_processor": {"device": str(dev)}})
    anchor = torch.empty(0, device=dev)
    image_policy = SimpleNamespace(config=cfg, parameters=lambda: iter([anchor]))

    def preprocess(batch):
        return PI05Policy._preprocess_images(image_policy, batch)

    weights = load_pi05_weights(repo, dev, steps=int(cfg.num_inference_steps))
    return GiPi05Loop(cfg, pre, post, preprocess, weights, dev)
