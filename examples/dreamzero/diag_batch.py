#!/usr/bin/env python3
"""Discriminating diagnostic for the CFG-batch delta: is it my batching or the kernels?

Case A (cond,cond): batch two IDENTICAL rows (cond context + cond cache twice). Row 0 must equal
the stock cond output bit-for-bit if the batched execution is semantically faithful — identical
data per row removes every mixed-batch kernel excuse.
Case B: also compare row 0 vs row 1 of that same batch (self-consistency).

Then the batched-arm latency that OOM'd in the verify (12-chunk session, GPU alloc headroom).
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/dreamzero-repo")
sys.path.insert(0, "/home/ubuntu/dreamzero-repo/eval_utils")
sys.path.insert(0, "/home/ubuntu/InstinctFlash/examples/dreamzero")

from verify_cfg_batch import build_wrapper, session_obs, run_session  # noqa: E402
from cfg_batch import install_cfg_batch  # noqa: E402

OUT = Path("/home/ubuntu/iwm_distill/bench_dreamzero_h100")


def main():
    import types
    wrapper = build_wrapper()
    head = wrapper._policy.trained_model.action_head
    stock_method = head._run_diffusion_steps

    calls = session_obs(41, "pick up the banana and place it in the bowl", 1)

    # Reference: stock, capture first denoise-step cond output
    cap = {}
    def capture_stock(self, **kw):
        out = stock_method(**kw)
        if "x" not in cap and kw.get("action") is not None:
            cap["x"] = (out[0][0].float().clone(), out[0][1].float().clone())
        return out
    head._run_diffusion_steps = types.MethodType(capture_stock, head)
    run_session(wrapper, calls)
    ref = cap.pop("x")
    head._run_diffusion_steps = stock_method
    torch.cuda.empty_cache()

    # Case A: batched with (cond, cond) — duplicate context AND duplicate cond cache
    install_cfg_batch(head)
    batched = head._run_diffusion_steps
    def condcond(self, **kw):
        if kw.get("action") is not None and "x" not in cap:
            kw2 = dict(kw)
            kw2["context"] = [kw["context"][0], kw["context"][0]]
            kw2["kv_caches"] = [kw["kv_caches"][0], kw["kv_caches"][0]]
            kw2["crossattn_caches"] = [kw["crossattn_caches"][0], kw["crossattn_caches"][0]]
            self._ifl_kv_marker = None                      # force stacked rebuild from cond,cond
            out = batched(**kw2)
            cap["x"] = (out[0][0].float().clone(), out[0][1].float().clone())
            cap["row1"] = (out[1][0].float().clone(), out[1][1].float().clone())
            self._ifl_kv_marker = None                      # don't poison later state
            raise _Done()
        return stock_method(**kw)
    class _Done(Exception):
        pass
    head._run_diffusion_steps = types.MethodType(condcond, head)
    try:
        run_session(wrapper, calls)
    except _Done:
        pass
    head._run_diffusion_steps = stock_method
    got, row1 = cap.pop("x"), cap.pop("row1")

    d_row0 = float((ref[0] - got[0]).abs().max()), float((ref[1] - got[1]).abs().max())
    d_rows = float((got[0] - row1[0]).abs().max()), float((got[1] - row1[1]).abs().max())
    print(f"condcond row0 vs stock cond: video {d_row0[0]:.3e} action {d_row0[1]:.3e}")
    print(f"condcond row0 vs row1      : video {d_rows[0]:.3e} action {d_rows[1]:.3e}")

    # Batched latency, 12-chunk session
    head._ifl_cfg_batch_installed = False
    head._ifl_kv_marker = None
    install_cfg_batch(head)
    torch.cuda.empty_cache()
    calls = session_obs(31, "pick up the banana and place it in the bowl", 6)
    wrapper.reset({}); wrapper._current_session_id = None
    lat = []
    wrapper.infer(dict(calls[0]))
    for obs in calls[1:]:
        t0 = time.perf_counter()
        wrapper.infer(dict(obs))
        lat.append((time.perf_counter() - t0) * 1000)
    lat = lat[2:]
    res = {"label": "dz_ours_cfg_batched", "wall_ms_p50": round(statistics.median(lat), 1),
           "wall_ms_mean": round(statistics.mean(lat), 1),
           "diag_condcond_row0_vs_stock": d_row0, "diag_row0_vs_row1": d_rows}
    print(json.dumps(res, indent=1))
    (OUT / "ours.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
