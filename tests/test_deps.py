#!/usr/bin/env python3
"""Does the tracer derive what we previously had to remember?

Three dependencies bit this engine, each found by hand after a wrong number. This asserts the
tracer finds all three by itself:

  1. host mutation inside the region      (the ring bookkeeping -> max|d| 1.398)
  2. (start, count) as graph key fields   (omitted -> 6 graphs, stale replays)
  3. the cross-attention K/V as a read    (rebuilt per episode -> nan on episode 2)

    python tests/test_deps.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
LINGBOT_ROOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT_ROOT, "wan_va"))
sys.path.insert(0, LINGBOT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "lingbot_va_robotwin"))

import torch
import trace_block
from trace_block import DIM, HEADS, TEXT_LEN

from instinctflash.planners.deps import derive_signature
from instinctflash.passes.lingbot.ring_kv import RingKVAddressing

DEV, DT, KV, B, NL, N = torch.device("cuda"), torch.bfloat16, 2048, 2, 3, 32


class _M:
    def __init__(self, blocks):
        self.blocks = blocks

    def named_parameters(self):
        for i, b in enumerate(self.blocks):
            for n, p in b.named_parameters():
                yield f"block{i}.{n}", p

    def named_buffers(self):
        for i, b in enumerate(self.blocks):
            for n, p in b.named_buffers():
                yield f"block{i}.{n}", p


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0

    RingKVAddressing().install(None, type("S", (), {"_reset": lambda s, prompt=None: None}))

    blocks = []
    for _ in range(NL):
        blk = trace_block.build_block(DEV, DT)
        blk.attn1.init_kv_cache("pos", KV, HEADS, DIM // HEADS, DEV, DT, B)
        blocks.append(blk)
    model = _M(blocks)

    h = torch.randn(B, N, DIM, device=DEV, dtype=DT)
    enc = torch.randn(B, TEXT_LEN, DIM, device=DEV, dtype=DT)
    temb = torch.randn(B, N, 6, DIM, device=DEV, dtype=DT)
    rot = torch.randn(1, N, 1, DIM // HEADS // 2, device=DEV, dtype=torch.complex64)

    def stack():
        x = h
        for b in blocks:
            x = b(x, enc, temb, rot, update_cache=0, cache_name="pos")
        return x

    ring = blocks[0].attn1.attn_caches["pos"]["_ring"]
    rings = [b.attn1.attn_caches["pos"]["_ring"] for b in blocks]

    def set_all(fieldname):
        def _set(v):
            for r in rings:
                r[fieldname] = v
        return _set

    from instinctflash.adapters.lingbot import state_roots
    nr = state_roots(model)
    nr.update({"in:hidden": h, "in:encoder": enc, "in:temb": temb, "in:rot": rot})
    sig = derive_signature(
        stack, name_roots=nr,
        roots=[b.attn1.attn_caches for b in blocks],
        host_fields={"ring.start": lambda: ring["start"], "ring.count": lambda: ring["count"],
                     "ring.next_id": lambda: ring["next_id"]},
        perturb={"ring.start": set_all("start"), "ring.count": set_all("count"),
                 "ring.next_id": set_all("next_id")})
    print(sig)

    ok = True
    print("\n--- did it find what we previously had to remember? ---")

    found_mut = any("next_id" in m or "_ring" in m for m in sig.host_mutations)
    ok &= found_mut
    print(f"  {'OK  ' if found_mut else 'FAIL'} (1) host mutation inside the region: "
          f"{list(sig.host_mutations)[:3]}")

    keys = set(sig.key_fields)
    found_keys = {"ring.start", "ring.count"} <= keys
    ok &= found_keys
    print(f"  {'OK  ' if found_keys else 'FAIL'} (2) start+count derived as graph key fields: "
          f"{sorted(keys)}")

    no_false_write = "in:encoder" not in sig.writes and "in:rot" not in sig.writes
    ok &= no_false_write
    print(f"  {'OK  ' if no_false_write else 'FAIL'} read-only inputs are NOT reported as writes "
          f"(views alias storage; the schema says who mutates)")

    kv_reads = [r for r in sig.reads if r.startswith("kv[")]
    found_kv = len(kv_reads) >= NL
    ok &= found_kv
    print(f"  {'OK  ' if found_kv else 'FAIL'} (3) KV pools appear as external reads: "
          f"{len(kv_reads)} buffers, e.g. {kv_reads[:3]}")

    cap, why = sig.capturable()
    ok &= not cap                      # with commit inline, it must REFUSE
    print(f"  {'OK  ' if not cap else 'FAIL'} refuses capture while commit is inline: {why}")

    print("\n--- and with deferred commit, it should allow capture ---")
    type(blocks[0].attn1)._iwm_defer_commit = True
    sig2 = derive_signature(stack, name_roots=state_roots(model),
                            roots=[b.attn1.attn_caches for b in blocks])
    cap2, why2 = sig2.capturable()
    ok &= cap2
    print(f"  {'OK  ' if cap2 else 'FAIL'} {why2}")
    kv_writes = [w for w in sig2.writes if w.startswith("kv[")]
    ok &= len(kv_writes) >= NL
    print(f"  {'OK  ' if len(kv_writes) >= NL else 'FAIL'} KV pool writes ARE tracked: "
          f"{len(kv_writes)} buffers, e.g. {kv_writes[:3]}")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
