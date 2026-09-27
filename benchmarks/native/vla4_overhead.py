"""Split GI VLA4 per-call time into set_prompt, input staging and graph replay."""
import sys
import time

import torch


def main():
    from benchmarks.native.vla4_parity import _paths
    from instinctflash.engines.vla4.engine import Vla4Engine
    from instinctflash.engines.vla4.weights import load_vla4_weights
    dev = torch.device("cuda")
    data = torch.load(sys.argv[1])
    eng = Vla4Engine(load_vla4_weights(_paths()[0], dev), dev)
    recs = data["records"]
    for r in recs[:3]:
        eng.set_prompt(r["lang_tokens"][r["lang_masks"].bool()].numpy())
        eng.infer(r["vision"].to(dev), r["state"].to(dev), r["noise"][0].to(dev))
    torch.cuda.synchronize()
    for r in recs:
        ids = r["lang_tokens"][r["lang_masks"].bool()].numpy()
        v, st, n = r["vision"].to(dev), r["state"].to(dev), r["noise"][0].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); eng.set_prompt(ids); torch.cuda.synchronize()
        t1 = time.perf_counter(); eng.infer(v, st, n); torch.cuda.synchronize()
        t2 = time.perf_counter(); eng._graphs.run(eng._rows, eng._forward); torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(f"set_prompt {1e3*(t1-t0):6.2f} ms  infer {1e3*(t2-t1):6.2f} ms  bare replay {1e3*(t3-t2):6.2f} ms")


if __name__ == "__main__":
    main()
