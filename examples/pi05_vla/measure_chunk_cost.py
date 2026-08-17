#!/usr/bin/env python3
"""What a pi05 control cycle actually costs, and why the obvious number is the wrong one.

    HF_TOKEN=... python examples/pi05_vla/measure_chunk_cost.py

Measured on one H100, 105 consecutive predict() calls in a single episode:

    call   1    557.6 ms      <- includes warm-up
    call  51    345.9 ms
    call 101    319.8 ms
    the other 102 calls: median 2.49 ms

    chunk period 50 steps | one chunk inference 333 ms | 6.7 ms/step amortised | ~150 Hz equivalent

READ THIS BEFORE QUOTING A pi05 LATENCY. A naive timing loop reports "warm median 3 ms", and that is
not inference -- it is a QUEUE POP. pi05 predicts a 50-step action chunk, and `select_action` serves
49 of every 50 calls out of that buffer, so 97% of the samples never touch the model. Reporting 3 ms
as pi05's inference latency would be off by 100x, and it would be off in the flattering direction.

Both numbers are true and they answer different questions. 2.49 ms is what a controller waits on a
typical tick. 333 ms is what it waits on every fiftieth tick, and it is the only number an
optimization can move: eleven transformer forwards, one prefix plus ten flow-matching steps.

The chunk boundary is visible at calls 1, 51 and 101, so `chunk_size: 50` is read out of the timings
rather than taken on trust from the checkpoint's config.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import instinctwm                                                  # noqa: E402
from pi05_iwm.adapter import Pi05Adapter                           # noqa: E402

BASE = "lerobot/pi05_base"
CALLS = 105
PROMPT = "Put the exhaust fans back to the slots."
SLOW_MS = 50.0          # anything above this is a chunk inference, not a dequeue


def declare(into: Path) -> Path:
    """A checkpoint that carries pi05's weights BY REFERENCE, so nothing is copied to run this."""
    into.mkdir(parents=True, exist_ok=True)
    (into / "config.json").write_text("{}")
    (into / "instinctwm.json").write_text(json.dumps({
        "instinctwm_schema": 1,
        "execution": {"model_id": "example-org/pi05", "backbone": "pi05", "servable": True,
                      "guidance": {"action": "none"},          # flow matching, no CFG
                      "nfe": {"action": 10, "prefix": 1},
                      "base_weights": BASE},
    }, indent=2))
    return into


def main() -> int:
    if "pi05" not in instinctwm.available_models():
        instinctwm.register("pi05", Pi05Adapter)

    with tempfile.TemporaryDirectory() as td:
        runtime = instinctwm.Runtime.from_pretrained(declare(Path(td) / "pi05"))
        obs = runtime.observation.example()
        obs["prompt"] = PROMPT

        times = []
        with runtime, runtime.episode(prompt=PROMPT) as episode:
            for _ in range(CALLS):
                t0 = time.perf_counter()
                episode.predict(obs)
                times.append((time.perf_counter() - t0) * 1000)

    slow = [(i, t) for i, t in enumerate(times) if t > SLOW_MS]
    fast = [t for t in times if t <= SLOW_MS]
    print(f"{CALLS} predict() calls, the ones over {SLOW_MS:.0f} ms:")
    for i, t in slow:
        print(f"   call {i + 1:4}   {t:7.1f} ms")
    print(f"dequeued calls: {len(fast)}, median {statistics.median(fast):.2f} ms")

    if len(slow) < 2:
        print("\nFewer than two chunk boundaries seen -- raise CALLS above the chunk size.")
        return 1
    period = slow[1][0] - slow[0][0]
    # skip call 1: it carries warm-up, and averaging it in would understate steady-state cost
    infer = statistics.median([t for _, t in slow[1:]])
    print(f"\nchunk period      {period} steps   (read from the timings, not the config)")
    print(f"chunk inference   {infer:.0f} ms      <- the number to optimize")
    print(f"amortised         {infer / period:.1f} ms/step   ~{1000 * period / infer:.0f} Hz equivalent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
