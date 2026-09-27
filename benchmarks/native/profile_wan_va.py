"""In-graph op profile of one GI LingBot-VA forward (video B=2 T=240 or action B=1 T=32)."""
import argparse

import torch

from benchmarks.native.graph_op_profile import profile_engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("video", "action"), default="video")
    p.add_argument("--history", type=int, default=2, help="committed cycles before profiling")
    args = p.parse_args()
    import instinctflash.engines.wan_va.engine as E
    from benchmarks.native.wan_va_parity import script, transformer_dir
    from instinctflash.engines.wan_va.weights import load_wan_weights
    dev = torch.device("cuda")
    gi = E.GiWanTransformer(load_wan_weights(transformer_dir(), dev), dev, pool_slots=36 * 240 + 36 * 32)
    gi.create_empty_cache("pos", 72, 240, 32, dev, torch.bfloat16, 2)
    steps = script(args.history + 1, dev)
    # Run the history cycles, then stage one forward of the requested kind and profile its graph.
    cut = [i for i, s in enumerate(steps) if s[1] == args.history][0]
    for name, c, i, inp, uc, act in steps[:cut]:
        if name == "clear_pred":
            gi.clear_pred_cache("pos")
        else:
            gi(inp, update_cache=uc, cache_name="pos", action_mode=act)
    name, c, i, inp, uc, act = next(s for s in steps[cut:] if s[0] == args.mode)
    gi(inp, update_cache=uc, action_mode=act)          # compile + capture this key
    lat = inp["noisy_latents"]
    B = 1 if act else lat.shape[0]
    T = (lat.shape[2] * lat.shape[3] * lat.shape[4]) if act else lat.shape[2] * (lat.shape[3] // 2) * (lat.shape[4] // 2)
    key = (act, B, T)
    names = ["gemm", "attention", "residual_norm_quant", "row_quant", "wan_qkv"]
    profile_engine(gi, {args.mode: lambda: gi._forward(key)}, E, names)


if __name__ == "__main__":
    main()
