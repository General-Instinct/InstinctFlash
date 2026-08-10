"""The InstinctWM adapter for `gridworld-ar`, written from OUTSIDE InstinctWM.

AUDIT RECORD. Every InstinctWM name this file has to import or implement is a thing an external
model author must learn. Counted at the bottom of the module. Nothing in the InstinctWM tree was
modified to make this work.
"""

from __future__ import annotations

from instinctwm import AdapterSpec, PhaseSpec

from gridworld_wm.model import GridworldAR, quantize


class _Impl:
    """What `build_in_process` returns. The runtime calls `reset` and `predict` on this.

    NOTE FOR THE AUDIT: nothing in InstinctWM's documented `BackendAdapter` protocol says this
    object exists or what shape it has. I found the required methods by reading the error message
    from `InProcessBackend`, which is a better failure mode than silence but is not documentation.
    """

    def __init__(self, model, history: int):
        import torch
        self._torch, self._model, self._history = torch, model, history
        self._tokens: list[int] = []

    def reset(self, **conditioning) -> None:
        # This model has no prompt conditioning. It ignores kwargs rather than rejecting them,
        # because the runtime forwards whatever the caller passed to `reset`.
        self._tokens = [0]

    def predict(self, observation):
        if not self._tokens:
            self.reset()
        self._tokens.append(quantize(observation.get("obs", observation)))
        window = self._tokens[-self._history:]
        t = self._torch.tensor([window], dtype=self._torch.long)
        with self._torch.no_grad():
            action = self._model(t)
        return {"action": action.squeeze(0).cpu().numpy()}

    def close(self) -> None:
        self._model = None


class GridworldAdapter:
    """Implements the parts of InstinctWM's adapter contract that are actually required."""

    BACKBONE = "gridworld_ar"

    def spec(self) -> AdapterSpec:
        # ONE phase, ONE forward, no streams, no guidance, no purity assertions, no commit.
        return AdapterSpec(
            model_id="gridworld-ar",
            param_bytes=0,
            streams=(),
            phases=(PhaseSpec(name="act", nfe=1),),
            guidance={},
        )

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        import json
        from pathlib import Path

        import torch
        from safetensors.torch import load_file

        pkg = Path(checkpoint.path)
        cfg = json.loads((pkg / "config.json").read_text())
        model = GridworldAR(**cfg)
        model.load_state_dict(load_file(str(pkg / "model.safetensors")))
        model.eval()
        if device:
            model.to(device)
        return _Impl(model, cfg["history"])


#: Registered on import of the package -- see __init__.py.
ADAPTER = GridworldAdapter
