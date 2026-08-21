#!/usr/bin/env python3
"""pi05 through the whole platform path: declaration -> adapter -> plan -> real actions.

    HF_TOKEN=... python examples/pi05_vla/run_pi05_end_to_end.py

Real weights (`lerobot/pi05_base`), real flow-matching inference, public API only. This is the test
that pi05 support exists, as opposed to pi05 support type-checking.

WHAT IT TOOK TO GET HERE, because each wall is a thing a reader will hit too:

  1  `check_whether_transformers_replace_is_installed_correctly()` asserts
     `transformers.__version__ == "4.53.2"` exactly, and raises "An incorrect transformer version is
     used" on anything else.
  2  pi05's preprocessor needs `relative_actions_processor`, absent from lerobot 0.4.4's registry.
  3  `google/paligemma-3b-pt-224` is gated, so the VLM backbone 401s without an accepted licence.
  4  The checkpoint ships `device_processor: {"device": "cpu"}`, which lands tokens on the CPU while
     the weights are on cuda:0. Overridden at load, or every forward raises.
  5  The processor calls `state.cpu().numpy()`, so a numpy observation dies on `.cpu()`. The adapter
     converts; the DECLARED contract stays numpy, because that is what a robot stack has.

1-3 are environment, and lerobot 0.6.1 on Python 3.13 fixes all three. 4-5 are the adapter's job and
are handled in `pi05_iwm/adapter.py`.

Expected on one H100: load ~1.3 s, first predict ~600 ms, action (32,) float32, finite, varying.
The timings here are NOT pi05's per-step latency -- see `measure_chunk_cost.py` for why.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                 # noqa: E402
import instinctflash                                                  # noqa: E402
from pi05_iwm.adapter import Pi05Adapter                           # noqa: E402

BASE = "lerobot/pi05_base"
PROMPT = "Put the exhaust fans back to the slots."
FAILED: list[str] = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(label)
    return cond


def declare(into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    (into / "config.json").write_text("{}")
    (into / "instinctflash.json").write_text(json.dumps({
        "instinctflash_schema": 1,
        "execution": {"model_id": "example-org/pi05-declared", "backbone": "pi05", "servable": True,
                      "guidance": {"action": "none"},
                      "nfe": {"action": 10, "prefix": 1},
                      "base_weights": BASE},
        # provenance never reaches the runtime; it is here to prove that stays true
        "provenance": {"training_method": "not the runtime's business"},
    }, indent=2))
    return into


def main() -> int:
    if "pi05" not in instinctflash.available_models():
        instinctflash.register("pi05", Pi05Adapter)

    with tempfile.TemporaryDirectory() as td:
        pkg = declare(Path(td) / "pi05")

        print("1. describe() -- no weights downloaded")
        d = instinctflash.describe(pkg)
        print(f"   backbone={d['backbone']}  nfe={d['nfe']}")
        check(d["backbone"] == "pi05", "declares the pi05 backbone")
        check("not the runtime's business" not in json.dumps(d), "provenance does not leak")

        print("2. Runtime.from_pretrained() -- resolves the adapter, plans, places")
        t0 = time.time()
        runtime = instinctflash.Runtime.from_pretrained(pkg)
        print(f"   loaded in {time.time() - t0:.1f} s")
        check(runtime.plan is not None, "compiled a plan")
        print(f"   expects: {runtime.observation.describe()}")

        print("3. closed loop, real actions")
        obs = runtime.observation.example()
        obs["prompt"] = PROMPT
        times, actions = [], []
        with runtime, runtime.episode(prompt=PROMPT) as episode:
            for _ in range(6):
                t0 = time.perf_counter()
                out = episode.predict(obs)
                times.append((time.perf_counter() - t0) * 1000)
                actions.append(np.asarray(out["action"] if isinstance(out, dict) else out))

    a = actions[0]
    print(f"   first {times[0]:.0f} ms, then {min(times[1:]):.0f}-{max(times[1:]):.0f} ms "
          f"(dequeued from the chunk -- not inference)")
    print(f"   action {a.shape} {a.dtype}  {[round(float(v), 4) for v in a.ravel()[:5]]}")
    check(a.size > 0 and bool(np.isfinite(a).all()), "actions are finite and non-empty")
    check(float(np.abs(np.stack(actions)).std()) > 1e-6, "actions advance through the chunk")

    print()
    if FAILED:
        print(f"FAILED {len(FAILED)}: {FAILED}")
        return 1
    print("PASS: pi05 loads from a declaration and produces real actions through the public API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
