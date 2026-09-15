"""The exact decoded first camera frames used by the historical SDE1 screen."""
from pathlib import Path

from cosmos3_sde1.overlay import digest, require


FIXTURE_SHA256 = "37843e22fa6dd9a2abf2bae390ccb8e5c4319446e3d3fab5d0d477860065f411"


def load_frames(path):
    import numpy as np
    from PIL import Image

    path = Path(path)
    require(digest(path) == FIXTURE_SHA256, "use the pinned non-pickle recorded fixture")
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"]
        require(frames.dtype == np.uint8 and frames.shape == (13, 3, 240, 320, 3),
                "safe fixture geometry differs")
        # Index0 is the separate initial-history frame; historical jpeg_0 starts at index1.
        return [np.asarray(Image.fromarray(frame).resize((640, 540))) for frame in frames[1:13, 0]]
