"""InstinctWM adapter for LeRobot's ACT family. Facts from lerobot/act_* config.json.

ACT is the degenerate case on purpose: an encoder-decoder transformer that emits a 100-action chunk
in ONE forward. No denoise loop, no guidance, no cache carried between control steps. If the
abstraction is really about execution rather than about diffusion, it has to describe this too.
"""
from __future__ import annotations

from instinctwm import AdapterSpec, GuidanceRule, PhaseSpec
from instinctwm.adapters.base import GuidanceMode

BACKBONE = "act"


class ACTAdapter:
    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id="lerobot/act_aloha_sim_transfer_cube_human",
            param_bytes=206_000_000,
            streams=(),                 # nothing persists across control steps
            phases=(PhaseSpec(name="action", nfe=1),),   # ONE forward. No refinement loop at all.
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            purity=(),                  # no loop, so nothing to hoist out of one
            obs_decode_modules=(),
            notes={"family": "act", "chunk_size": "100", "n_obs_steps": "1"},
        )

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None):
        """Load the upstream LeRobot policy named by `execution.base_weights`.

        The impl contract is `predict` plus optional `reset`. LeRobot's ACTPolicy already matches it
        almost exactly: `select_action` hides an internal action-chunk queue and returns one action
        per call, and `reset` clears that queue at episode boundaries. That correspondence is the
        interesting part -- action-chunk buffering is a real execution concept, both systems arrived
        at hiding it behind one verb, and nothing model-specific had to reach the runtime.
        """
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy

        repo = (checkpoint.execution.extra or {}).get("base_weights")
        if not repo:
            raise RuntimeError(
                f"{checkpoint.model_id}: no local weights and no execution.base_weights, so there "
                f"is nothing to load. Declare the upstream repo id in base_weights.")
        policy = ACTPolicy.from_pretrained(repo)
        policy.eval()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy.to(dev)
        return _ACTLoop(policy, dev)


class _ACTLoop:
    """One control cycle over an ACT policy. No commit phase: nothing persists between cycles."""

    def __init__(self, policy, device):
        import torch
        self._torch, self._p, self._dev = torch, policy, device

    def reset(self, **conditioning) -> None:
        self._p.reset()                      # drops the buffered action chunk

    def predict(self, observation):
        batch = {k: v for k, v in observation.items() if k.startswith("observation.")}
        batch = {k: self._as_tensor(v) for k, v in batch.items()}
        with self._torch.no_grad():
            action = self._p.select_action(batch)
        return {"action": action.squeeze(0).cpu().numpy()}

    def _as_tensor(self, v):
        t = v if isinstance(v, self._torch.Tensor) else self._torch.as_tensor(v)
        if t.dtype != self._torch.float32:
            t = t.float()
        return t.to(self._dev)

    def close(self) -> None:
        self._p = None
