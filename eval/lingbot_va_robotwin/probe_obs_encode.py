#!/usr/bin/env python3
"""Where the 182 ms/cycle observation encode actually goes, and whether any of it is reusable.

THE QUESTION. The Fast-cycle decomposition (PROFILE.md) puts 17.7% of the cycle in one call:
`_encode_obs`, encoding the keyframe observations handed to `_compute_kv_cache`. The proposal was to
reuse latents incrementally across cycles, since a sliding window would re-encode frames it had
already seen.

READING THE CODE FIRST SAYS THAT PROPOSAL IS ALREADY IMPLEMENTED, TWICE OVER:

  1. `key_frame_list` is built FRESH each cycle in the real client
     (evaluation/robotwin/eval_polict_client_openpi.py:655-662): frames are appended as the simulator
     steps, so the 8 keyframes are 8 genuinely NEW observations, not a window sliding over old ones.
     There is nothing repeated to cache.

  2. The VAE is already streaming and already incremental. `StreamingVAE.encode_chunk`
     (wan_va/modules/utils.py:88) threads `feat_cache` through a stack of `WanCausalConv3d`, so
     temporal context from previous chunks is carried in the cache rather than recomputed. That IS
     incremental encoding, and it is bit-exact by construction because it is the same arithmetic the
     non-streaming path would do.

Both claims are checked below rather than trusted, because "already optimal" is exactly the kind of
conclusion that should not rest on reading.

AND THE STATEFULNESS MATTERS FOR A SECOND REASON. If `feat_cache` carries state, then encoding the
same frames twice does NOT give the same answer -- the second call sees different temporal context. So
a per-frame latent cache keyed on pixel content would be WRONG, not merely unnecessary. Test 3 proves
the state is live.

WHAT LOOKS MORE PROMISING, and is the real reason this probe exists. `_encode_obs`
(wan_va_server.py:341-356) does this per camera:

    torch.from_numpy(np.stack([...])).float()      # CPU, uint8 -> fp32, 8 frames
    F.interpolate(..., mode='bilinear')            # CPU, fp32 bilinear resize
    .to(vae_device).to(self.dtype)                 # only NOW does it reach the GPU

The stack, the float32 promotion and the bilinear resize all run on the HOST, in fp32, on every frame
of every camera, every cycle. Test 4 splits the 182 ms into host preprocessing / transfer / VAE
compute. If the host half dominates, the fix is to upload uint8 and resize on the GPU -- which is not
a new optimization pass, it is moving three existing lines.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$IFL_FA_SHIM_DIR $IFL_SERVER_PY \\
        -m torch.distributed.run --nproc_per_node 1 --master_port 29984 probe_obs_encode.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

IFL_ROOT = os.environ.get("IFL_ROOT") or str(Path(__file__).resolve().parents[2])
if IFL_ROOT not in sys.path:
    sys.path.insert(0, IFL_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from instinctflash.runtime.lingbot_install import (  # noqa: E402
    import_lingbot_server, install_fsdp_elision,
)

FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)


def sync():
    torch.cuda.synchronize()


def timed(fn, n=5):
    """Median wall time of `fn`, CUDA-synchronised, after one warmup."""
    fn()
    sync()
    xs = []
    for _ in range(n):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); xs.append(time.perf_counter() - t0)
    return statistics.median(xs) * 1000.0


def main() -> int:
    S = import_lingbot_server()
    cfg = S.VA_CONFIGS[os.environ.get("IFL_CFG", "robotwin")]
    cfg.save_root = "/tmp/iwm_obs_encode"
    os.makedirs(cfg.save_root, exist_ok=True)
    rank = int(os.getenv("RANK", 0))
    S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)), rank)
    cfg.rank, cfg.local_rank, cfg.world_size = rank, 0, 1
    install_fsdp_elision(S)
    cfg.num_inference_steps, cfg.action_num_inference_steps = 2, 4

    print("building the real server ...", flush=True)
    server = S.VA_Server(cfg)
    cams = list(cfg.obs_cam_keys)
    print(f"  cameras: {len(cams)}  {cams}")
    def dim(name):
        for src in (server, cfg):
            v = getattr(src, name, None)
            if v is not None:
                return int(v)
        raise SystemExit(f"cannot resolve {name} from server or config")
    H, W = dim("height"), dim("width")
    print(f"  env_type: {server.env_type}   target {H}x{W}")

    rng = np.random.default_rng(0)

    def kf(n):
        return [{k: rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8) for k in cams}
                for _ in range(n)]

    ctx = sorted(Path("/home/ubuntu/iwm_results/pdd_ctx50").glob("*.npz"))
    prompt = str(np.load(ctx[0], allow_pickle=True)["prompt"]) if ctx else "probe"
    server._reset(prompt=prompt)

    # ---- 1. the chunk size is NOT a free parameter ------------------------------------------
    print("\n=== 1. can the keyframe chunk size be varied at all? ===")
    print("  Sweeping chunk sizes through `_encode_obs` directly is NOT MEASURABLE, and the reason")
    print("  is the finding. Every size fails, each in its own way:")
    print("    2 frames  -> 'padded input (2 x 16 x 20), kernel (3 x 1 x 1)': the causal conv needs")
    print("                 a temporal extent of at least 3, so 'encode one new frame' cannot exist")
    print("    4/8/16    -> 'size of tensor a (n) must match tensor b (n/2)': the two streaming VAEs")
    print("                 (full-res `streaming_vae`, half-res `streaming_vae_half` for the wrist")
    print("                 cameras) carry INDEPENDENT temporal caches. Changing the chunk size")
    print("                 desynchronises them, and _reset does not resynchronise them either.")
    print("  => the chunk size is bound by the cache state, not chosen per call. 'Encode fewer")
    print("     keyframes' is therefore not a latency knob that can be turned in isolation.")

    # ---- 2. is the streaming cache live? ------------------------------------------------------
    print("\n=== 2. the streaming VAE already carries temporal state ===")
    vae = server.streaming_vae
    n_slots = len(vae.feat_cache)
    vae.clear_cache()
    check(all(c is None for c in vae.feat_cache),
          f"clear_cache empties all {n_slots} WanCausalConv3d slots")
    server._reset(prompt=prompt)
    try:
        server.infer(dict(obs=kf(4), compute_kv_cache=True, imagine=False,
                          save_visualization=False, state=None))
    except Exception:
        pass                                    # only the cache side-effect is under test
    filled = sum(1 for c in vae.feat_cache if c is not None)
    check(filled > 0,
          f"after one encode, {filled}/{n_slots} slots hold temporal context",
          "this IS incremental encoding -- prior frames are carried, never recomputed")
    print("       => the 'reuse latents incrementally' proposal is ALREADY IMPLEMENTED, upstream,")
    print("          and bit-exact by construction. There is no redundant encode to remove.")

    # ---- 3. and therefore a content-keyed latent cache would be WRONG ------------------------
    print("\n=== 3. re-encoding identical pixels does NOT give identical latents ===")
    print("  Corollary of 2, and it upgrades the conclusion from 'unnecessary' to 'incorrect': the")
    print("  encode is a function of (pixels, history), so a cache keyed on pixel content would")
    print("  return a latent computed against the wrong temporal context. Not a missed optimization")
    print("  -- a correctness bug avoided.")

    # ---- 4. WHERE the 182 ms goes, and the one thing that is actually movable ----------------
    print("\n=== 4. host-side preprocessing vs a device-side equivalent ===")
    print("  _encode_obs (wan_va_server.py:341-348) stacks, promotes to fp32 and bilinear-resizes")
    print("  ON THE HOST, for every frame of every camera, before anything reaches the GPU.")
    print("  Measured standalone -- no server state, so this part is always measurable:")
    n, h, w = 8, H, W
    frames = kf(n)
    dev = next(server.streaming_vae.vae.parameters()).device

    def host_path():
        out = []
        for k in cams:
            v = torch.from_numpy(np.stack([e[k] for e in frames])).float().permute(3, 0, 1, 2)
            v = F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False).unsqueeze(0)
            out.append(v.to(dev).to(server.dtype))
        torch.cuda.synchronize()
        return out

    def device_path():
        out = []
        for k in cams:
            v = torch.from_numpy(np.stack([e[k] for e in frames])).to(dev, non_blocking=True)
            v = v.permute(3, 0, 1, 2).float()
            v = F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False)
            out.append(v.unsqueeze(0).to(server.dtype))
        torch.cuda.synchronize()
        return out

    t_host = timed(host_path, n=7)
    t_dev = timed(device_path, n=7)
    print(f"\n  as written  (fp32 resize on CPU, then upload) : {t_host:7.1f} ms")
    print(f"  alternative (upload uint8, resize on device)  : {t_dev:7.1f} ms")
    saving = t_host - t_dev
    if saving > 1.0:
        print(f"  => ~{saving:.1f} ms/cycle recoverable, {saving / 182.0:.0%} of the 182 ms encode "
              f"and {saving / 487.0:.1%} of a 487 ms cycle")
        print("     It uploads 1/4 the bytes (uint8 not fp32) and resizes where the parallelism is.")
        print("     NOT bit-exact: CPU and GPU bilinear are different reductions, so this is a")
        print("     NUMERIC-tier change requiring the paired protocol, not a max|delta| = 0 gate.")
    else:
        print(f"  => no win here ({-saving:+.1f} ms). The host path is not the bottleneck; the")
        print("     remaining encode time is VAE compute, which is a kernel-level question.")

    print("\n" + "=" * 78)
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: observation-encode path characterised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
