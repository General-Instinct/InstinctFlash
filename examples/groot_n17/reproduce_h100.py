#!/usr/bin/env python3
"""Rerun the README GR00T-N1.7-3B H100 pair: upstream eager vs the Runtime DEFAULT arm.

Protocol, exactly as the published row (verify_fastpaths.py): DROID embodiment, synthetic
observations built from the checkpoint's own modality config (video (1, 2, 256, 256, 3),
seeded per case), timing p50 of 15 calls after 3 warmup on the obs0 case. The stock arm is
NVIDIA's ``Gr00tPolicy.get_action`` eager. The ours arm is ``Runtime.from_pretrained`` on this
package with no flags: what the family DEFAULT serves — fast decode, the backbone fastpath,
and the DiT CUDA graphs, self-check gated.

Quality gate, run here end to end: the six fixed-seed cases (two prompt switches) compared
action-for-action between the two arms; the family tier is BITEXACT, so the gate is exact
equality (max |d| == 0.0).

    examples/groot_n17/reproduce_h100.sh          # wraps this with the venv/GPU knobs

Env:
    GR00T_N17_CHECKPOINT   weights dir for both arms (default: the HF cache snapshot)
    GR00T_ROOT             Isaac-GR00T checkout (the upstream policy stack)
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("GR00T_ROOT", str(Path.home() / "Isaac-GR00T")))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))   # repo root: the package imports instinctflash

README_PAIR = (114.8, 59.1)
STATE_DIMS = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
SHORT = "pick up the object"
LONG = ("carefully pick up the leftmost small red block and place "
        "it inside the open drawer on the right side of the table")
VIDEO_BASES = ("exterior_image_1_left", "wrist_image_left")


def _skip_transformers_mistral_hub_probe():
    import transformers.tokenization_utils_base as tub

    def _no_mistral_patch(cls, tokenizer, *args, **kwargs):
        return tokenizer

    tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(_no_mistral_patch)


def snapshot() -> str:
    if os.environ.get("GR00T_N17_CHECKPOINT"):
        return os.environ["GR00T_N17_CHECKPOINT"]
    hub = os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots")
    return os.path.join(hub, os.listdir(hub)[0])


def state_value(base):
    v = np.zeros((1, 1, STATE_DIMS[base]), dtype=np.float32)
    if base == "eef_9d":
        v[0, 0, 3:9] = [1, 0, 0, 0, 1, 0]   # identity frame; zeros break the rot6d SVD
    return v


def make_video(seed):
    rng = np.random.default_rng(seed)
    return {k: rng.integers(0, 256, size=(1, 2, 256, 256, 3), dtype=np.uint8)
            for k in VIDEO_BASES}


def make_cases():
    cases = [(f"obs{s}", make_video(100 + s), SHORT, 1234 + s) for s in (0, 1, 2, 3)]
    fixed = make_video(0)
    cases.append(("new-prompt", fixed, LONG, 4321))
    cases.append(("new-prompt-2", fixed, LONG, 4322))
    return cases


def rekey(values_by_base, keys):
    return {k: values_by_base[k.split(".")[-1]] for k in keys}


def flatten(action):
    if isinstance(action, tuple):
        action = action[0]
    if not isinstance(action, dict):
        return {"action": np.asarray(action)}
    return {k.split(".")[-1]: np.asarray(v.cpu() if torch.is_tensor(v) else v)
            for k, v in action.items()}


def run_arm(predict_case, cases):
    outs, timing_obs = [], None
    for name, video, prompt, seed in cases:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        outs.append((name, predict_case(video, prompt)))
        if name == "obs0":
            timing_obs = (video, prompt)
    for _ in range(3):
        predict_case(*timing_obs)
    torch.cuda.synchronize()
    lat = []
    for _ in range(15):
        t0 = time.perf_counter()
        predict_case(*timing_obs)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return outs, statistics.median(lat)


def main() -> int:
    _skip_transformers_mistral_hub_probe()
    cases = make_cases()

    # ---- stock arm: NVIDIA's policy, eager -------------------------------------------------
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
                         model_path=snapshot(), device="cuda:0")
    cfg = policy.modality_configs
    state_by_base = {b: state_value(b) for b in STATE_DIMS}

    def stock_case(video, prompt):
        obs = {
            "video": rekey(video, tuple(cfg["video"].modality_keys)),
            "state": rekey(state_by_base, tuple(cfg["state"].modality_keys)),
            "language": {k: [[prompt]] for k in cfg["language"].modality_keys},
        }
        return flatten(policy.get_action(obs))

    stock_outs, stock_p50 = run_arm(stock_case, cases)
    print(f"stock (Gr00tPolicy eager)               p50 = {stock_p50:.1f} ms")

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    # ---- ours arm: the Runtime DEFAULT, no flags --------------------------------------------
    from instinctflash import Runtime

    runtime = Runtime.from_pretrained(HERE, device="cuda:0", placement="in_process")
    current = {"prompt": None}

    def ours_case(video, prompt):
        if prompt != current["prompt"]:
            runtime.reset(prompt=prompt)
            current["prompt"] = prompt
        # Provide both bare and config-prefixed keys; the adapter reads its config's exact
        # names from the nested dicts and ignores extras.
        v = dict(video)
        v.update({f"video.{k}": a for k, a in video.items()})
        s = dict(state_by_base)
        s.update({f"state.{k}": a for k, a in state_by_base.items()})
        return flatten(runtime.predict({"video": v, "state": s})["actions"])

    ours_outs, ours_p50 = run_arm(ours_case, cases)
    stats = dict(getattr(runtime._backend._impl, "backend_stats", {}) or {})
    print(f"ours (Runtime default arm)              p50 = {ours_p50:.1f} ms   "
          f"[{stats.get('backend')}, fast_decode={stats.get('fast_decode')}, "
          f"backbone_fastpath={stats.get('backbone_fastpath')}, "
          f"graphs={stats.get('graph_captures')}]")
    runtime.close()

    # ---- gate: exact equality (the family tier is BITEXACT) ---------------------------------
    worst = 0.0
    for (name, s), (_, o) in zip(stock_outs, ours_outs):
        d = max(float(np.abs(s[k] - o[k]).max()) for k in s)
        worst = max(worst, d)
        print(f"  {name:14} max |d| = {d:.3e}")
    ok = worst == 0.0
    print(f"GATE ours vs stock: {'PASS (bitexact)' if ok else f'delta {worst:.3e}'}")

    res = {
        "protocol": "verify_fastpaths.py protocol: obs0 timing p50 of 15 x 3 warmup; "
                    "6 gate cases, exact equality",
        "stock_ms_p50": round(stock_p50, 1),
        "ours_ms_p50": round(ours_p50, 1),
        "speedup": round(stock_p50 / ours_p50, 2),
        "gate_max_abs_d": worst,
        "backend_stats": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                              else str(v)) for k, v in stats.items()},
        "readme_pair_ms": list(README_PAIR),
        "device": torch.cuda.get_device_name(0),
    }
    print(json.dumps(res, indent=1))
    (HERE / "reproduce_h100_results.json").write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
