#!/usr/bin/env python3
"""Episode-mode latency: N consecutive control cycles, ONE reset, ring never rewound.

WHY THIS EXISTS

`probe_latency` resets between repeats. That rewinds the ring KV interval to (0, 0), so every
repeat replays the same sequence of graph keys the first repeat captured -- and the first repeat is
the one the protocol discards. Any cost that recurs per cycle because the ring advances is
therefore invisible: it gets paid once, in the run we throw away.

That is not hypothetical. Measured in-process with no resets, the graph pass captured ~7.9 new
graphs EVERY cycle and never converged, because the graph key contains the ring signature
(start, count) and the pool grows 272 tokens per cycle. ~85% of a cycle was capture. `probe_latency`
reported 1171.9 ms for the same build.

So this probe does what an episode does: reset once, then run. A RoboTwin episode is ~53 cycles and
the 9792-slot pool saturates around cycle 36, so the pre/post-saturation split is reported
separately -- the steady state after saturation is the number that describes long-horizon control.

    python probe_episode.py --port 29058 --cycles 45 [--server-log /path/to/server.log]
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/RoboTwin")
from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

CAMS = ["observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"]
SLOTS_PER_CYCLE = 272          # video + action tokens committed per control cycle


def make_obs(rng, h=240, w=320):
    return {k: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for k in CAMS}


def read_captures(path):
    """captures/replays after each _infer, from the server's own stats line."""
    if not path:
        return []
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return []
    pat = (r"captures=(\d+) replays=(\d+) unique_keys=(\d+) held=(\d+) evicted=(\d+)")
    rows = [tuple(int(x) for x in m) for m in re.findall(pat, txt)]
    if rows:
        return rows
    # older stats line without unique_keys/held/evicted
    return [(int(a), int(b), 0, 0, 0) for a, b in
            re.findall(r"captures=(\d+) replays=(\d+)", txt)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29058)
    ap.add_argument("--cycles", type=int, default=45)
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--kv-slots", type=int, default=9792)
    ap.add_argument("--prompt", default="Use the left arm to lift the plastic drink bottle head-up")
    ap.add_argument("--server-log", default=None,
                    help="optional: parse the server's capture stats and align them to cycles")
    args = ap.parse_args()

    sat = args.kv_slots // SLOTS_PER_CYCLE
    c = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"episode mode: {args.cycles} cycles, ONE reset, ring never rewound")
    print(f"KV pool {args.kv_slots} slots / {SLOTS_PER_CYCLE} per cycle -> saturates ~cycle {sat}")

    before = len(read_captures(args.server_log))
    c.infer(dict(reset=True, prompt=args.prompt, save_visualization=False))
    rng = np.random.default_rng(0)
    first = make_obs(rng)

    times = []
    for i in range(args.cycles):
        t0 = time.perf_counter()
        a = c.infer(dict(obs=first, prompt=args.prompt, save_visualization=False))["action"]
        nkf = args.keyframes // 2 if i == 0 else args.keyframes
        kfs = [make_obs(rng) for _ in range(nkf)]
        c.infer(dict(obs=kfs, compute_kv_cache=True, imagine=False,
                     save_visualization=False, state=a))
        times.append((time.perf_counter() - t0) * 1e3)

    caps = read_captures(args.server_log)[before:]
    per_cycle_caps = []
    prev = None
    for i in range(min(len(caps), len(times))):
        cur = caps[i][0]
        per_cycle_caps.append(cur - prev if prev is not None else cur)
        prev = cur

    print(f"\n{'cyc':>4} {'ms':>9} {'new caps':>9} {'uniq keys':>10} {'held':>6} "
          f"{'evicted':>8} {'cum replays':>12}")
    for i, t in enumerate(times):
        nc = per_cycle_caps[i] if i < len(per_cycle_caps) else None
        row = caps[i] if i < len(caps) else None
        if row is None:
            print(f"{i:4d} {t:9.1f} {'-':>9} {'-':>10} {'-':>6} {'-':>8} {'-':>12}")
        else:
            _c, rp, uk, hd, ev = row
            print(f"{i:4d} {t:9.1f} {('-' if nc is None else nc):>9} {uk:10d} {hd:6d} "
                  f"{ev:8d} {rp:12d}")

    pre = [t for i, t in enumerate(times) if i < sat]
    post = [t for i, t in enumerate(times) if i >= sat]
    print("\n" + "=" * 62)
    if pre:
        print(f"pre-saturation  (cycles 0-{sat-1}) : mean {np.mean(pre):8.1f} ms  "
              f"min {np.min(pre):7.1f}  max {np.max(pre):7.1f}")
    if post:
        print(f"post-saturation (cycles {sat}+)   : mean {np.mean(post):8.1f} ms  "
              f"min {np.min(post):7.1f}  max {np.max(post):7.1f}")
    print(f"whole episode                   : mean {np.mean(times):8.1f} ms")
    if per_cycle_caps:
        tot = sum(x for x in per_cycle_caps if x is not None)
        steady = per_cycle_caps[sat:] if len(per_cycle_caps) > sat else []
        hit = 1.0 - (tot / max(1, caps[-1][1])) if caps else float("nan")
        uk, hd, ev = caps[-1][2], caps[-1][3], caps[-1][4]
        print(f"unique graph keys: {uk}   held at end: {hd}   evictions: {ev}")
        print(f"captures: {tot} over the episode, {np.mean(per_cycle_caps):.1f}/cycle"
              + (f", {np.mean(steady):.1f}/cycle after saturation" if steady else ""))
        print(f"graph cache hit rate (1 - captures/replays): {hit:.3%}")
        if steady and np.mean(steady) > 0.5:
            print("\nVERDICT: the graph key is NOT stable over an episode -- captures continue")
            print("         indefinitely. Any speedup measured with a resetting probe overstates")
            print("         what a real episode sees.")
        elif steady:
            print("\nVERDICT: the graph key stabilizes; the cache converges within an episode.")
    else:
        print("(pass --server-log to also report capture counts and hit rate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
