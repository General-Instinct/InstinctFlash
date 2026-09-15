"""Checkpoint-native pi05 processing around the Thor FP8 action generator.

The engine receives processed images and state-containing token IDs. It returns
normalized model actions; the checkpoint processor owns robot-space decoding.
"""
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


class Pi05EngineLoop:
    def __init__(self, config, pre, post, preprocess_images, frontend_factory, device):
        import torch
        self.config = config
        self._torch = torch
        self._pre, self._post = pre, post
        self._preprocess_images = preprocess_images
        self._frontend_factory = frontend_factory
        self._device = torch.device(device)
        self._frontends = {}
        self._queue = deque()
        self._prompt = ""
        self._action_dim = int(config.output_features["action"].shape[0])
        if not 1 <= self._action_dim <= 32:
            raise ValueError("pi05 engine requires an action dimension between 1 and 32")
        if not 1 <= config.n_action_steps <= config.chunk_size:
            raise ValueError("pi05 n_action_steps must be within the computed action horizon")

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device=None, action_steps=None):
        import torch
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor

        repo = str(Path(checkpoint.path))
        dev = torch.device(device or "cuda")
        if dev.type != "cuda":
            raise ValueError("pi05 Thor engine requires a CUDA device")
        cfg = PI05Config.from_pretrained(repo)
        cfg.device = str(dev)
        cfg.compile_model = False
        cfg.validate_features()
        if action_steps is not None:
            cfg.num_inference_steps = int(action_steps)
        if cfg.num_inference_steps != 10:
            raise ValueError("pi05 engine currently requires the checkpoint's ten-step schedule")
        if (cfg.max_action_dim != 32 or cfg.paligemma_variant != "gemma_2b"
                or cfg.action_expert_variant != "gemma_300m"):
            raise ValueError("pi05 Thor engine requires gemma_2b/gemma_300m with max_action_dim=32")
        if getattr(cfg, "rtc_config", None) is not None:
            raise ValueError("pi05 engine action queue does not yet support RTC")
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=repo,
            preprocessor_overrides={"device_processor": {"device": str(dev)}},
            postprocessor_overrides={"device_processor": {"device": str(dev)}})
        # The upstream image routine needs only config and a parameter device;
        # do not construct a second multi-billion-parameter model to reuse it.
        device_anchor = torch.empty(0, device=dev)
        image_policy = SimpleNamespace(config=cfg, parameters=lambda: iter([device_anchor]))

        def preprocess(batch):
            return PI05Policy._preprocess_images(image_policy, batch)

        def frontend(views):
            with torch.cuda.device(dev):
                return Pi05TorchFrontendThor(repo, num_views=views, use_fp8=True,
                    action_chunk=int(cfg.chunk_size), action_output="normalized", autotune=0)

        return cls(cfg, pre, post, preprocess, frontend, dev)

    def reset(self, **conditioning):
        self._prompt = str(conditioning.get("prompt") or conditioning.get("task") or "")
        self._queue.clear()

    def predict(self, observation, *, executed_action=None):
        torch = self._torch
        batch = {}
        for key, value in observation.items():
            if not key.startswith("observation."):
                continue
            value = value if torch.is_tensor(value) else torch.as_tensor(value)
            if value.dtype not in (torch.float32, torch.uint8):
                value = value.float()
            batch[key] = value.to(self._device)
        batch["task"] = str(observation.get("prompt") or self._prompt)
        with torch.no_grad():
            batch = self._pre(batch)
            if not self._queue:
                self._refill(batch)
            action = self._post(self._queue.popleft())
        action = action if torch.is_tensor(action) else torch.as_tensor(action)
        return {"action": action.squeeze(0).detach().cpu().numpy()}

    def _refill(self, batch):
        torch = self._torch
        images, masks = self._preprocess_images(batch)
        active_images = []
        for image, mask in zip(images, masks):
            if image.shape[0] != 1 or mask.numel() != 1:
                raise ValueError("pi05 Runtime currently serves one observation at a time")
            if not bool(mask.item()):
                continue
            image = image[0]
            if image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            active_images.append(image.to(torch.float16).cpu().numpy())
        if not 1 <= len(active_images) <= 3:
            raise ValueError("pi05 Thor engine requires one to three active cameras")
        tokens = batch["observation.language.tokens"]
        mask = batch["observation.language.attention_mask"].bool()
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape != mask.shape:
            raise ValueError("pi05 token IDs and masks must have matching [1, length] shapes")
        ids = tokens[mask].detach().cpu().numpy()
        if not 1 <= len(ids) <= 256:
            raise ValueError("pi05 Thor engine requires 1..256 active prompt tokens")
        count = len(active_images)
        # Missing cameras are masked by the native processor. Their masked KV
        # entries can be omitted; remaining camera order follows that processor.
        if count not in self._frontends:
            self._frontends[count] = self._frontend_factory(count)
        engine = self._frontends[count]
        with torch.cuda.device(self._device) if self._device.type == "cuda" else nullcontext():
            engine.set_prompt(ids)
            raw = engine.infer({"images": active_images})["actions"]
        if raw.shape != (self.config.chunk_size, 32):
            raise RuntimeError(f"pi05 engine returned unexpected action shape {raw.shape}")
        actions = torch.as_tensor(raw[:, :self._action_dim], device=self._device)
        if not torch.isfinite(actions).all():
            raise RuntimeError("pi05 engine returned nonfinite actions")
        self._queue.extend(actions[:self.config.n_action_steps].unsqueeze(1))

    def close(self):
        self._queue.clear()
        self._frontends.clear()
        self._pre = self._post = self._preprocess_images = self._frontend_factory = None
