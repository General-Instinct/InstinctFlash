#!/usr/bin/env python3
"""Gates for the CFG-batched DreamZero arm, in-process on one GPU.

Order of evidence, per the house discipline:
  0. null control — stock vs stock on identical sessions. If the model's own reruns differ,
     BITEXACT is off the table for ANY serving of it and the batched deltas are judged against
     this envelope (the LingBot-VLA-V2 lesson).
  1. batched vs stock on the same sessions — captured-equivalent case plus unseen cases
     (different observation bytes, different session lengths, and a different prompt).
  2. in-process latency, stock vs batched, 12 calls each after 3 warmup.

A session = first call (1 frame/cam, warms causal cache) + N chunk calls (4 frames/cam).
Noise is deterministic (action head generates from a fixed seed), so sessions replay exactly.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ubuntu/dreamzero-repo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval_utils"))
sys.path.insert(0, "/home/ubuntu/InstinctFlash/examples/dreamzero")

OUT_DIR = Path("/home/ubuntu/iwm_distill/bench_dreamzero_h100")
SNAP = next(Path("/home/ubuntu/.cache/huggingface/hub/models--GEAR-Dreams--DreamZero-DROID/snapshots").iterdir())


def build_wrapper():
    import os
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29821")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    from serve_dreamzero_wan22 import DreamZeroWan225BPolicy, _get_expected_video_resolution
    policy = GrootSimPolicy(embodiment_tag=EmbodimentTag("oxe_droid"), model_path=str(SNAP),
                            tokenizer_path_override=None, device="cuda", device_mesh=device_mesh)
    h, w = _get_expected_video_resolution(policy)
    return DreamZeroWan225BPolicy(groot_policy=policy, image_height=h, image_width=w)


def session_obs(seed, prompt, n_chunks):
    rng = np.random.default_rng(seed)
    calls = []
    for i in range(n_chunks + 1):
        nf = 1 if i == 0 else 4
        calls.append({
            "observation/exterior_image_0_left": rng.integers(0, 256, size=(nf, 160, 320, 3), dtype=np.uint8),
            "observation/exterior_image_1_left": rng.integers(0, 256, size=(nf, 160, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 256, size=(nf, 160, 320, 3), dtype=np.uint8),
            "prompt": prompt,
            "session_id": f"s{seed}",
        })
    return calls


def run_session(wrapper, calls):
    wrapper.reset({})
    wrapper._current_session_id = None
    outs = []
    for obs in calls:
        outs.append(np.asarray(wrapper.infer(dict(obs))))
    return outs


def max_delta(a_list, b_list):
    return max(float(np.abs(a - b).max()) for a, b in zip(a_list, b_list))


def main():
    from cfg_batch import install_cfg_batch
    wrapper = build_wrapper()
    head = wrapper._policy.trained_model.action_head
    stock_method = head._run_diffusion_steps

    def to_stock():
        head._run_diffusion_steps = stock_method
        head._ifl_cfg_batch_installed = False
        head._ifl_kv_marker = None
        head._ifl_kv2 = head._ifl_ca2 = None
        torch.cuda.empty_cache()

    def to_batched():
        head._ifl_cfg_batch_installed = False
        head._ifl_kv_marker = None
        install_cfg_batch(head)
        torch.cuda.empty_cache()

    report = {}

    # 0. null control
    calls = session_obs(7, "pick up the banana and place it in the bowl", 2)
    to_stock()
    a = run_session(wrapper, calls)
    b = run_session(wrapper, calls)
    report["null_stock_vs_stock"] = max_delta(a, b)
    print(f"null control (stock vs stock): {report['null_stock_vs_stock']:.3e}")

    # 0b. single-forward delta: how much does ONE batched DiT forward differ before the
    # multistep scheduler amplifies it? Run one 1-chunk session where only the FIRST computed
    # denoise step's outputs are compared via a hook.
    single_deltas = []
    orig_run = stock_method
    captured = {}
    def capturing(self, **kw):
        out = orig_run(**kw)
        if "first" not in captured and kw.get("action") is not None:
            captured["first"] = (out[0][0].float().clone(), out[0][1].float().clone() if out[0][1] is not None else None)
        return out
    import types as _t
    calls1 = session_obs(41, "pick up the banana and place it in the bowl", 1)
    to_stock(); head._run_diffusion_steps = _t.MethodType(capturing, head)
    run_session(wrapper, calls1); ref_first = captured.pop("first")
    head._run_diffusion_steps = stock_method
    to_batched()
    batched_method = head._run_diffusion_steps
    def capturing2(self, **kw):
        out = batched_method(**kw)
        if "first" not in captured and kw.get("action") is not None:
            captured["first"] = (out[0][0].float().clone(), out[0][1].float().clone() if out[0][1] is not None else None)
        return out
    head._run_diffusion_steps = _t.MethodType(capturing2, head)
    run_session(wrapper, calls1); got_first = captured.pop("first")
    head._run_diffusion_steps = stock_method
    d_video = float((ref_first[0] - got_first[0]).abs().max()) if ref_first[0].shape == got_first[0].shape else f"shape {tuple(ref_first[0].shape)} vs {tuple(got_first[0].shape)}"
    d_action = (float((ref_first[1] - got_first[1]).abs().max())
                if (ref_first[1] is not None and got_first[1] is not None and ref_first[1].shape == got_first[1].shape) else None)
    report["single_forward_delta"] = {"video": d_video, "action": d_action}
    print(f"single-forward delta: video {d_video:.3e} action {d_action}")

    # 1. batched vs stock, 6 cases
    cases = [(7, "pick up the banana and place it in the bowl", 2),
             (11, "pick up the banana and place it in the bowl", 2),
             (12, "pick up the banana and place it in the bowl", 3),
             (13, "pick up the banana and place it in the bowl", 1),
             (21, "open the top drawer and put the block inside", 2),
             (22, "open the top drawer and put the block inside", 3)]
    deltas = []
    for seed, prompt, n in cases:
        calls = session_obs(seed, prompt, n)
        to_stock()
        ref = run_session(wrapper, calls)
        torch.cuda.empty_cache()
        to_batched()
        got = run_session(wrapper, calls)
        d = max_delta(ref, got)
        deltas.append(d)
        print(f"case seed={seed} prompt={prompt[:20]!r} chunks={n}: max|d| = {d:.3e}")
    report["batched_vs_stock_deltas"] = deltas

    # 2. latency, in-process
    calls = session_obs(31, "pick up the banana and place it in the bowl", 20)
    def timed(label):
        run_session(wrapper, calls[:4])          # warm
        wrapper.reset({}); wrapper._current_session_id = None
        lat = []
        wrapper.infer(dict(calls[0]))
        for obs in calls[1:16]:
            t0 = time.perf_counter()
            wrapper.infer(dict(obs))
            lat.append((time.perf_counter() - t0) * 1000)
        lat = lat[3:]
        r = {"label": label, "wall_ms_p50": round(statistics.median(lat), 1),
             "wall_ms_mean": round(statistics.mean(lat), 1)}
        print(json.dumps(r))
        return r

    to_stock()
    report["stock_inproc"] = timed("dz_stock_inprocess")
    to_batched()
    report["ours_inproc"] = timed("dz_ours_cfg_batched")

    report["speedup"] = round(report["stock_inproc"]["wall_ms_p50"] /
                              report["ours_inproc"]["wall_ms_p50"], 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "verify_and_ours.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if "inproc" not in k}, indent=1))


if __name__ == "__main__":
    main()
