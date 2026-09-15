"""Renderer determinism probe — is the LIBERO env's first frame a function of the seed?

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python render_probe.py <out.json> [--save obs.npz]

Spawns the env twice in FRESH subprocesses (matching how eval jobs see it), resets with
the same seed, and compares (a) the physics state arrays, (b) the rendered pixels,
bit-wise. Measured on this box (H100, EGL): physics is bit-identical, pixels are not
(~2.3% of pixels differ, max |d| ~73/255) — MuJoCo EGL offscreen rendering is not
process-deterministic. This is why the T3 null control is a same-obs replay
(null_replay.py) rather than a closed-loop rerun.

--save also writes the first process's obs to an npz (the replay control's input).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

CHILD = r"""
import sys, hashlib, json
import numpy as np
from lerobot.envs.factory import make_env, make_env_config
cfg = make_env_config("libero", task="libero_spatial", task_ids=[0])
envs = make_env(cfg, n_envs=1, use_async_envs=False)
env = envs["libero_spatial"][0]
obs, info = env.reset(seed=[1000], options={})
np.savez(sys.argv[1], image=obs["pixels"]["image"], image2=obs["pixels"]["image2"])
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "/"))
        else:
            out[p + k] = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:12]
    return out
open(sys.argv[2], "w").write(json.dumps(flat(obs.get("robot_state") or {})))
"""


def main() -> int:
    import numpy as np
    out_json = sys.argv[1]
    save_npz = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None

    with tempfile.TemporaryDirectory() as td:
        runs = []
        for i in (0, 1):
            npz, st = f"{td}/o{i}.npz", f"{td}/s{i}.json"
            subprocess.run([sys.executable, "-c", CHILD, npz, st],
                           check=True, capture_output=True, env=os.environ)
            runs.append((npz, json.load(open(st))))
        state_identical = runs[0][1] == runs[1][1]
        a, b = np.load(runs[0][0]), np.load(runs[1][0])
        pixels = {}
        for k in a.files:
            x, y = a[k].astype(int), b[k].astype(int)
            d = np.abs(x - y)
            pixels[k] = {"identical": bool((d == 0).all()),
                         "pct_differing_px": round(100 * float((d > 0).mean()), 3),
                         "max_abs_delta": int(d.max()),
                         "mean_abs_delta": round(float(d.mean()), 4)}
        if save_npz:
            import shutil
            shutil.copy(runs[0][0], save_npz)

    result = {"what": "two fresh processes, env.reset(seed=[1000]), first obs compared bitwise",
              "physics_state_identical": state_identical,
              "pixels": pixels,
              "verdict": ("renderer nondeterministic (physics deterministic)"
                          if state_identical and not all(p["identical"] for p in pixels.values())
                          else "bit-deterministic" if state_identical else "PHYSICS DIVERGED")}
    json.dump(result, open(out_json, "w"), indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
