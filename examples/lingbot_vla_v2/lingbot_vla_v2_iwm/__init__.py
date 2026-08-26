"""InstinctFlash adapter for the published LingBot-VLA-V2 RobotWin policy."""

__all__ = ["LingBotVLAV2Adapter"]


def __getattr__(name):
    # Importing e.g. ``lingbot_vla_v2_iwm.static_capture`` from FlashRT must not pull the
    # Runtime/core package into a serving-only environment. The adapter entry point imports
    # its module directly, so a lazy convenience export keeps both packages independent.
    if name == "LingBotVLAV2Adapter":
        from .adapter import LingBotVLAV2Adapter

        return LingBotVLAV2Adapter
    raise AttributeError(name)
