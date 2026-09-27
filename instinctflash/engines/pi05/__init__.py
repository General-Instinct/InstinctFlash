"""GI pi05 engine (PaliGemma prefix + Gemma action expert, per-token FP8)."""


def build_pi05_engine_loop(checkpoint, *, device=None, action_steps=None):
    from instinctflash.engines.pi05.loop import build_pi05_engine_loop as build
    return build(checkpoint, device=device, action_steps=action_steps)
