"""T3-cert null controls over the live server — run BEFORE any paired episode.

    /home/ubuntu/tools/pi05env313/bin/python null_replay.py <assets.json> <calib.npz> <out.json>

Two gates, one server lifetime (the same lifetime the paired arm B will use):

GATE 1 — same-obs replay (engine determinism over the network, pilot pattern): two passes
of [reset(seed=7); 28 x infer(same frame, same state-bearing prompt ids)]; per-chunk
sha256 over float32 action bytes must match pairwise across passes and DIFFER within a
pass (the noise RNG must advance). Closed-loop bit-determinism stays undefined on this
sim stack (MuJoCo EGL renderer nondeterminism, measured in the pilot) — this replay is
the defined property.

GATE 2 — state-liveness + no-recapture (the state-blindness fix, verified): two prompts
with the SAME task text and DIFFERENT State suffixes (different token lengths, so two
different capture buckets). Checks:
  (a) liveness: reset(seed); infer(ids_a) vs reset(seed); infer(ids_b) — same noise, same
      pixels, different state text -> actions MUST differ (the pilot's arm B could not
      fail this way: it was state-blind by construction);
  (b) sanity: all actions finite and within a loose raw-normalized envelope;
  (c) no recapture: set_prompt_calls frozen across every infer, and per-infer server_ms
      stays at replay speed (a set_prompt would cost seconds, not milliseconds);
  (d) determinism across buckets: an A/B/A/B alternation replayed twice bit-matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remote_policy import EngineClient  # noqa: E402

HOST = os.environ.get("IFL_THOR_HOST", "100.68.159.80")
PORT = int(os.environ.get("IFL_THOR_PORT", "29930"))
N_CHUNKS = 28
SEED = 7


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> int:
    assets_json, calib_npz, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    assets = json.load(open(assets_json))
    ids_a = np.asarray(assets["replay"]["ids_a"], dtype=np.int64)
    ids_b = np.asarray(assets["replay"]["ids_b"], dtype=np.int64)
    z = np.load(calib_npz)                       # engine wire form fp16, frame 0 = task 0
    frame = {"image": z["image"][0], "wrist_image": z["wrist_image"][0]}

    client = EngineClient(HOST, PORT)
    meta = client.metadata

    def infer(ids):
        rep = client.call({"op": "infer", "prompt_ids": ids, **frame})
        a = np.ascontiguousarray(np.asarray(rep["actions"], dtype=np.float32))
        return a, rep

    # ---- GATE 1: same-obs replay -----------------------------------------------------
    def one_pass():
        client.call({"op": "reset", "seed": SEED})
        shas, svr = [], []
        for _ in range(N_CHUNKS):
            a, rep = infer(ids_a)
            shas.append(_sha(a))
            svr.append(rep.get("server_ms", 0.0))
        return shas, svr

    s1, v1 = one_pass()
    s2, v2 = one_pass()
    div = next((i for i, (x, y) in enumerate(zip(s1, s2)) if x != y), None)
    gate1 = {
        "what": "same-obs replay over ws://%s:%d — 2 passes x %d infers, reset seed %d, "
                "STATE-BEARING prompt ids (len %d)" % (HOST, PORT, N_CHUNKS, SEED, len(ids_a)),
        "identical": div is None,
        "first_divergence_chunk": div,
        "n_chunks": N_CHUNKS,
        "distinct_shas_within_pass": len(set(s1)),
        "rng_advances_within_pass": len(set(s1)) > 1,
        "server_ms_p50": sorted(v1 + v2)[len(v1 + v2) // 2],
        "pass1_shas": s1, "pass2_shas": s2,
    }

    # ---- GATE 2: state-liveness + no-recapture ----------------------------------------
    spc0 = client.call({"op": "ping"})["set_prompt_calls"]

    client.call({"op": "reset", "seed": SEED})
    a_first, rep_a = infer(ids_a)
    client.call({"op": "reset", "seed": SEED})
    b_first, rep_b = infer(ids_b)
    liveness = _sha(a_first) != _sha(b_first)

    def ab_pass():
        client.call({"op": "reset", "seed": SEED})
        shas, times = [], []
        for ids in (ids_a, ids_b, ids_a, ids_b):
            a, rep = infer(ids)
            shas.append(_sha(a))
            times.append(round(rep.get("server_ms", 0.0), 2))
        return shas, times

    ab1, t1 = ab_pass()
    ab2, t2 = ab_pass()
    spc1 = client.call({"op": "ping"})["set_prompt_calls"]

    sane = all(np.isfinite(x).all() and float(np.abs(x).max()) < 10.0
               for x in (a_first, b_first))
    all_ms = t1 + t2 + [round(rep_a.get("server_ms", 0.0), 2),
                        round(rep_b.get("server_ms", 0.0), 2)]
    gate2 = {
        "what": "two prompts, same task text, DIFFERENT State suffixes "
                "(len %d vs %d = two buckets)" % (len(ids_a), len(ids_b)),
        "prompt_a": assets["replay"]["prompt_a"],
        "prompt_b": assets["replay"]["prompt_b"],
        "state_liveness_actions_differ": bool(liveness),
        "actions_sane": bool(sane),
        "raw_abs_max": round(max(float(np.abs(a_first).max()),
                                 float(np.abs(b_first).max())), 4),
        "abab_replay_identical": ab1 == ab2,
        "abab_shas": [ab1, ab2],
        "set_prompt_calls_before": spc0,
        "set_prompt_calls_after": spc1,
        "no_recapture": spc0 == spc1,
        "per_infer_server_ms": all_ms,
        "server_ms_max": max(all_ms),
    }

    client.close()

    result = {
        "server": meta,
        "same_obs_replay": gate1,
        "state_liveness": gate2,
        "identical": gate1["identical"],
        "passed": bool(gate1["identical"] and gate1["rng_advances_within_pass"]
                       and liveness and sane and gate2["no_recapture"]
                       and gate2["abab_replay_identical"]),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(result, open(out_json, "w"), indent=1)
    print(f"null controls: replay identical={gate1['identical']} "
          f"liveness={liveness} no_recapture={gate2['no_recapture']} "
          f"abab_identical={gate2['abab_replay_identical']} sane={sane} "
          f"-> passed={result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
