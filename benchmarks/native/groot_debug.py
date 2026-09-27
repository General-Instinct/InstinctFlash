"""Stage-by-stage GI GR00T engine vs upstream head modules (same process, bundle record 0)."""
import sys

import torch


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max()), float((a - b).abs().mean() / b.abs().mean())


def main():
    from benchmarks.native.groot_parity import MODEL, _checkpoint_dir
    from instinctflash import Runtime
    from instinctflash.engines.groot_n17.engine import GrootN17Engine
    from instinctflash.engines.groot_n17.weights import load_groot_weights
    data = torch.load(sys.argv[1])
    r = data["records"][0]
    rt = Runtime.from_pretrained(MODEL, device="cuda:0", precision="native")
    rt.reset(prompt="x")
    head = rt._backend._impl._policy.model.action_head
    dev = torch.device("cuda")
    feats = r["features"].to(dev, torch.bfloat16)[None]
    with torch.no_grad():
        e_ref = head.vl_self_attention(head.vlln(feats))[0]
    eng = GrootN17Engine(load_groot_weights(_checkpoint_dir(), dev, r["embodiment"]), dev)
    eng.infer(r["features"].to(dev), r["image_mask"].to(dev), r["attention_mask"].to(dev),
              r["state"].to(dev), r["noise"][0].to(dev))
    S = feats.shape[1]
    print("VLSA out (max rel, mean rel):", rel(eng.vx[:S], e_ref))
    # cross K/V of block 0 (text) from the reference features
    blk = head.model.transformer_blocks[0]
    mask = (~r["image_mask"] & r["attention_mask"]).to(dev)
    with torch.no_grad():
        k_ref = blk.attn1.to_k(e_ref[mask])
    n = int(mask.sum())
    print("block0 text K:", rel(eng.kc[0, :n].reshape(n, -1), k_ref))
    # full action head with the reference features but the engine's (GI) DiT is not separable here;
    # compare end-to-end with upstream features fed into upstream DiT vs GI.
    with torch.no_grad():
        state = r["state"].to(dev, torch.bfloat16)
        emb = torch.tensor([r["embodiment"]], device=dev)
        s_ref = head.state_encoder(state, emb)
        print("state emb:", rel(eng.state_emb, s_ref[0]))


if __name__ == "__main__":
    main()
