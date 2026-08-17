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
