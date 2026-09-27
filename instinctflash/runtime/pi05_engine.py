"""Checkpoint-native pi05 processing around an FP8 action generator.

The engine receives processed images and state-containing token IDs. It returns
normalized model actions; the checkpoint processor owns robot-space decoding.
"""
from collections import deque


class Pi05EngineLoop:
    def __init__(self, config, pre, post, preprocess_images, device):
        import torch
        self.config = config
        self._torch = torch
        self._pre, self._post = pre, post
        self._preprocess_images = preprocess_images
        self._device = torch.device(device)
        self._frontends = {}
        self._queue = deque()
        self._prompt = ""
        self._action_dim = int(config.output_features["action"].shape[0])
        if not 1 <= self._action_dim <= 32:
            raise ValueError("pi05 engine requires an action dimension between 1 and 32")
        if not 1 <= config.n_action_steps <= config.chunk_size:
            raise ValueError("pi05 n_action_steps must be within the computed action horizon")

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
        """Run the action generator on one preprocessed batch and queue its actions."""
        raise NotImplementedError

    def close(self):
        self._queue.clear()
        self._frontends.clear()
        self._pre = self._post = self._preprocess_images = None
