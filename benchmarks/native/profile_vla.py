"""In-graph per-op profile of the GI VLA-4B / VLA-V2 engines on a parity bundle's first record.

    python -m benchmarks.native.profile_vla --model vla4 --bundle vla4_bundle.pt
"""
import argparse

import torch

from benchmarks.native.graph_op_profile import profile_engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("vla4", "vla2"), required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--sections", action="store_true", help="time sections without op markers")
    args = p.parse_args()
    dev = torch.device("cuda")
    r = torch.load(args.bundle)["records"][0]
    if args.model == "vla4":
        import instinctflash.engines.vla4.engine as E
        from benchmarks.native.vla4_parity import _paths
        from instinctflash.engines.vla4.weights import load_vla4_weights
        eng = E.Vla4Engine(load_vla4_weights(_paths()[0], dev), dev)
        eng.set_prompt(r["lang_tokens"][r["lang_masks"].bool()].numpy())
        eng.vis.copy_(r["vision"].reshape(192, -1))
        names = ["gemm", "attention", "residual_norm_quant", "row_quant", "gated_act_quant", "qkv_split"]
    else:
        import instinctflash.engines.vla2.engine as E
        from benchmarks.native.vla2_parity import _paths
        from instinctflash.engines.vla2.weights import load_vla2_weights
        eng = E.Vla2Engine(load_vla2_weights(_paths()[0], dev), dev)
        eng.set_prompt(r["lang_tokens"][r["lang_masks"].bool()].numpy())
        eng.vis.copy_(r["merged"])
        for i in range(3):
            eng.deep[i].copy_(r["deep"][i])
        names = ["gemm", "attention", "residual_norm_quant", "row_quant", "gated_act_quant", "qkv_split",
                 "moe_router", "moe_act_quant"]
    eng.state.copy_(r["state"].reshape(eng.state.shape))
    eng.noise.copy_(r["noise"].reshape(eng.noise.shape))
    M = eng._rows
    sections = {"prefix": lambda: eng._prefix(M), "expert": eng._expert}
    if args.sections:
        import time
        graphs = {}
        for name, fn in sections.items():
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs[name] = g
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            for g in graphs.values():
                g.replay()
        torch.cuda.synchronize()
        for name, g in graphs.items():
            ts = []
            for _ in range(15):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); g.replay(); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            ts.sort()
            print(f"{name:8s} p50 {ts[len(ts) // 2]:8.2f} ms")
        return
    profile_engine(eng, sections, E, names)


if __name__ == "__main__":
    main()
