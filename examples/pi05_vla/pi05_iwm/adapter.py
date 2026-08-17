"""InstinctWM adapter for LeRobot's pi05 VLA family. External plugin: no core changes.

Written to test whether InstinctWM's abstraction is real, using a model family deliberately unlike
LingBot-VA. Facts below come from lerobot/pi05_base's own config.json, not from guesses.
"""
from __future__ import annotations

from instinctwm import (AdapterSpec, GuidanceRule, KVLifetime, KVStreamSpec, PhaseSpec, PurityKey)
from instinctwm.adapters.base import GuidanceMode

BACKBONE = "pi05"


class Pi05Adapter:
    """A vision-language-action policy: one observation in, a 50-action chunk out."""

    def spec(self) -> AdapterSpec:
        return AdapterSpec(
            model_id="lerobot/pi05_base",
            param_bytes=14_467_165_872,
            # ONE stream, and its lifetime is the interesting part: the prefix K/V is recomputed
            # every control step (n_obs_steps=1, no history), so CHUNK -- not EPISODE like a WM.
            streams=(
                KVStreamSpec(name="prefix", tokens_per_frame=200, lifetime=KVLifetime.CHUNK),
            ),
            # vision+language prefix once, then 10 flow-matching steps over the action chunk.
            phases=(
                PhaseSpec(name="prefix", nfe=1, writes=frozenset({"prefix"})),
                PhaseSpec(name="action", nfe=10, reads=frozenset({"prefix"}),
                          truncatable=True, min_nfe=1, depends_on=("prefix",)),
            ),
            # Flow matching, no classifier-free guidance at all.
            guidance={"action": GuidanceRule(mode=GuidanceMode.NONE)},
            # The prefix is a pure function of the observation and prompt, constant across all 10
            # action steps -- the same shape of claim LingBot-VA makes at EPISODE scope.
            purity=(PurityKey(artifact="prefix_kv", fields=("images", "state", "prompt"),
                              scope=KVLifetime.CHUNK),),
            obs_decode_modules=(),      # a VLA predicts no pixels
            notes={"family": "vla", "chunk_size": "50", "n_obs_steps": "1"},
        )
