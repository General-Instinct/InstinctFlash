#!/usr/bin/env python3
"""Is the KV cache actually empty during the FIRST chunk's denoise?

The chunk-0 PDD reproduction rests on this. The static argument is tight -- `_compute_kv_cache`
reads `self.init_latent`, which only `_infer` ever sets (wan_va_server.py:447, initialised to None
at :382), so the observation ingest cannot precede the first denoise -- but "tight argument" is not
"measured", and a pre-seeded cache would silently make every training context wrong in a way the
loss curve would never reveal.

So: build the real server, reset it, and count occupied KV slots at three points --

    after _reset            expect 0
    inside every denoise    expect 0        (update_cache=0 writes slots then restore_cache()s them)
    after _compute_kv_cache expect > 0      (update_cache=2 is the real commit)

The middle one is the load-bearing measurement. It says the conditioning a chunk-0 training context
carries is exactly (observation, state, prompt) with no history term.

    IWM_FA_SHIM=1 CUDA_VISIBLE_DEVICES=0 $IWM_SERVER_PY probe_chunk0_cache.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from instinctwm.runtime.lingbot_install import import_lingbot_server  # noqa: E402


def occupancy(transformer, cache_name: str) -> tuple[int, int]:
    """(occupied slots, total slots) summed over every self-attention KV pool.

    The pools live on `WanAttention` (model.py:331), one per block -- NOT on the top-level
    transformer. An earlier version of this probe read `transformer.attn_caches`, got None
    everywhere, and printed a confident REFUTED. Reading the wrong object and reporting the
    resulting silence as evidence is worse than not measuring at all, so this walks the modules.
    """
    occ = tot = 0
    found = False
    for mod in transformer.modules():
        caches = getattr(mod, "attn_caches", None)
        if not isinstance(caches, dict) or caches.get(cache_name) is None:
            continue
        mask = caches[cache_name].get("mask")
        if mask is None:
            continue
        found = True
        occ += int(mask.sum().item())
        tot += int(mask.numel())
    return (occ, tot) if found else (-1, -1)


def synthetic_obs(cfg, n_frames: int = 4):
    """One observation in the shape the eval client sends: a list of per-frame camera dicts.

    Contents are random. This probe measures cache OCCUPANCY, not action quality, so pixel values
    are irrelevant -- what matters is that the tensors have shapes the real encoder accepts.

    n_frames must clear the streaming VAE's temporal kernel: `AutoencoderKLWan` uses a 3-tap time
    conv, so a single frame raises "Kernel size can't be greater than actual input size". The VAE
    also compresses time 4x, so 4 frames is the natural minimum chunk.
    """
    h, w = cfg.height, cfg.width
    frames = []
    for _ in range(n_frames):
        frames.append({k: np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                       for k in cfg.obs_cam_keys})
    state = torch.zeros(cfg.action_dim if hasattr(cfg, "action_dim") else 30,
                        1, cfg.action_per_frame)
    return {"obs": frames, "state": state.numpy()}


def main() -> int:
    S = import_lingbot_server()
    cfg_name = os.environ.get("IWM_CFG", "robotwin")
    cfg = S.VA_CONFIGS[cfg_name]          # same table the upstream main() uses (server.py:679)
    cfg.save_root = os.environ.get("IWM_PROBE_SAVE", "/tmp/iwm_probe_chunk0")
    os.makedirs(cfg.save_root, exist_ok=True)

    # Mirror wan_va_server.run(): the server reads rank/local_rank/world_size off the config, and
    # _configure_model shards only when a process group exists. Single rank here.
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    S.init_distributed(world_size, local_rank, rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, local_rank, world_size

    print("building the real server (this loads the 23 GB checkpoint) ...", flush=True)
    server = S.VA_Server(cfg)
    tr = server.transformer
    name = server.cache_name

    # -- instrument the denoise: record occupancy at every transformer call -------------------
    seen: list[tuple[str, int]] = []
    real_forward = tr.forward

    def probing_forward(input_dict, update_cache=0, cache_name="pos", action_mode=False, **kw):
        occ, _ = occupancy(tr, cache_name)
        seen.append((("action" if action_mode else "video"), occ, update_cache))
        return real_forward(input_dict, update_cache=update_cache, cache_name=cache_name,
                            action_mode=action_mode, **kw)

    tr.forward = probing_forward

    # Stub the VAE encode. This probe measures KV OCCUPANCY during the denoise, and the streaming
    # AutoencoderKLWan has strict temporal-chunk rules (3-tap time conv, 4x time compression, a
    # separate first frame) that have nothing to do with the question. Feeding it a correctly shaped
    # latent exercises the real transformer path while removing an irrelevant failure mode.
    lh, lw = ((cfg.height // 16) * 3) // 2, cfg.width // 16
    def _fake_encode(obs, *a, **k):
        return torch.randn(1, 48, 1, lh, lw, device=server.device, dtype=server.dtype)
    server._encode_obs = _fake_encode
    print(f"  (VAE encode stubbed: latent [1, 48, 1, {lh}, {lw}])")

    print("\n=== 1. after _reset ===")
    server._reset(prompt="Use the left arm to lift the plastic drink bottle head-up")
    occ, total = occupancy(tr, name)
    # -1 means the pool is not allocated yet, which is emptier than empty.
    print(f"  occupied {occ} / {total} slots" + ("   (pool not allocated yet)" if occ < 0 else ""))
    reset_empty = (occ <= 0)

    print("\n=== 2. during the first chunk's denoise (frame_st_id=0) ===")
    obs = synthetic_obs(cfg, n_frames=int(os.getenv("IWM_PROBE_FRAMES", "4")))
    server._infer(obs, frame_st_id=0)
    vid = [o for s, o, _ in seen if s == "video"]
    act = [o for s, o, _ in seen if s == "action"]
    print(f"  transformer calls: {len(seen)}  ({len(vid)} video, {len(act)} action)")
    print(f"  VIDEO  stream: occupancy min={min(vid)} max={max(vid)}")
    print(f"  ACTION stream: occupancy min={min(act)} max={max(act)}")
    commits = [(s, u) for s, _, u in seen if u != 0]
    print(f"  commits during the chunk: {commits}")
    # The scope under test is VIDEO FIRST. The action stream reads the video's committed KV, which
    # is a real coupling and exactly why it is a separate follow-up stage.
    denoise_empty = (max(vid) == 0)
    action_sees_history = bool(act) and max(act) > 0

    print("\n=== 3. after _compute_kv_cache (the real commit) ===")
    seen.clear()
    server._compute_kv_cache(obs)
    occ_after, _ = occupancy(tr, name)
    print(f"  occupied {occ_after} / {total} slots")
    commit_writes = (occ_after > 0)

    print("\n" + "=" * 68)
    ok = reset_empty and denoise_empty and commit_writes
    print(f"  reset leaves the cache empty                 : {reset_empty}")
    print(f"  chunk-0 VIDEO denoise sees an EMPTY cache    : {denoise_empty}   <-- the precondition")
    print(f"  chunk-0 ACTION denoise sees the video's KV   : {action_sees_history}")
    print(f"  the ingest commits slots afterwards          : {commit_writes}")
    print()
    if ok:
        print("CONFIRMED for the VIDEO stream: a chunk-0 video training context carries no")
        print("history term. Conditioning is exactly (observation, prompt).")
        if action_sees_history:
            print()
            print("NOT true for the action stream: it reads the KV the video stream commits on its")
            print("last denoise step, so an action-stream context is conditioned on the video")
            print("result. That coupling is why action distillation is a separate stage.")
    else:
        print("REFUTED: the chunk-0 video scope does not hold. Do NOT train on it.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
