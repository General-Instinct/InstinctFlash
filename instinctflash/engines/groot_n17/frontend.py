"""The action generator the GR00T N1.7 Runtime loop installs on the native action head."""
from __future__ import annotations

import torch

from instinctflash.engines.groot_n17.engine import ACT_DIM, HORIZON, GrootN17Engine
from instinctflash.engines.groot_n17.weights import embodiment_id, load_groot_weights


class GiGrootActionGenerator:
    """Replace ``action_head.get_action``: VLSA + DiT + decoder on the GI engine.

    The native state encoder still runs so the returned ``state_features`` keep their
    upstream meaning; RTC/inpainting requests fall back to the native DiT loop.
    """

    def __init__(self, head, engine, embodiment):
        config = head.config
        if (head.num_inference_timesteps != 4 or head.num_timestep_buckets != 1000
                or config.action_horizon != HORIZON or config.max_action_dim != ACT_DIM
                or config.state_history_length != 1 or not config.add_pos_embed
                or not config.use_alternate_vl_dit):
            raise ValueError("GR00T GI engine requires native 40x132 actions, one state frame and four steps")
        self.head, self.engine, self._embodiment_id = head, engine, int(embodiment)

    @classmethod
    def from_checkpoint(cls, head, checkpoint_dir, embodiment_tag, device):
        emb = embodiment_id(checkpoint_dir, embodiment_tag)
        return cls(head, GrootN17Engine(load_groot_weights(checkpoint_dir, device, emb), device), emb)

    def get_action(self, backbone_output, action_input, options=None):
        from transformers.feature_extraction_utils import BatchFeature
        state = action_input.state
        if tuple(state.shape) != (1, 1, ACT_DIM):
            raise ValueError("GR00T engine requires one normalized 132-dimensional state")
        emb = action_input.embodiment_id
        if emb.numel() != 1 or int(emb.item()) != self._embodiment_id:
            raise ValueError("GR00T observation embodiment differs from loaded engine weights")
        feats = backbone_output.backbone_features
        if feats.shape[0] != 1:
            raise ValueError("GR00T GI engine serves one observation at a time")
        with torch.no_grad():
            state_features = self.head.state_encoder(state, emb)
            if "action" in action_input:
                features = self.head.vl_self_attention(self.head.vlln(feats))
                return self.head.get_action_with_features(features, state_features, emb,
                                                          backbone_output, action_input, options)
            noise = torch.randn((1, HORIZON, ACT_DIM), dtype=self.head.dtype, device=feats.device)
            actions = self.engine.infer(feats[0], backbone_output.image_mask[0].bool(),
                                        backbone_output.backbone_attention_mask[0].bool(),
                                        state.float(), noise.float()[0])
            features = self.engine.vx[:feats.shape[1]].to(self.head.dtype)[None]
            return BatchFeature(data={"action_pred": actions.to(self.head.dtype)[None],
                                      "backbone_features": features, "state_features": state_features})
