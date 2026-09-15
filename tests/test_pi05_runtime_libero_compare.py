"""Reject corrupted evidence even when its outer artifact hashes are refreshed."""
import json

import numpy as np
import pytest

from benchmarks.vla.pi05_runtime_libero_compare import verify_episode, compare
from benchmarks.vla.util import ConfigurationError, sha256_file, sha256_json
from instinctflash.serving.msgpack_numpy import Packer


@pytest.fixture
def episode(tmp_path):
    identity = {"precision": "native"}
    manifest = {"identity": identity, "simulator_identity": {"revision": "test"}, "scene_sha256": "scene"}
    result = {**manifest, "ok": True, "success": False, "steps": 1, "task": 0, "seed": 0,
              "settle_steps": 10, "settle_action": [0, 0, 0, 0, 0, 0, -1], "camera_size": [360, 360]}
    (tmp_path / "episode.json").write_text(json.dumps(result))
    initial = {key: np.zeros(shape, dtype=np.uint8 if "image" in key else np.float64) for key, shape in {
        "agentview_image": (360, 360, 3), "robot0_eye_in_hand_image": (360, 360, 3),
        "robot0_eef_pos": (3,), "robot0_eef_quat": (4,), "robot0_gripper_qpos": (2,)}.items()}
    arrays = {"actions": np.zeros((1, 7), dtype=np.float32),
              **{"initial_" + k: v for k, v in initial.items()}}
    np.savez_compressed(tmp_path / "episode.npz", **arrays)
    acknowledgement = {"benchmark_seed": 0, "benchmark_identity_sha256": sha256_json(identity)}
    packets = [({**acknowledgement, "reset": True, "prompt": "move"}, acknowledgement.copy()),
               ({"libero_observation": initial, "prompt": "move"}, {"action": arrays["actions"][0]})]
    trace_root = tmp_path / "episode.trace"
    trace_root.mkdir()
    job = {k: result[k] for k in ("task", "seed", "success", "steps")}
    job["result"] = "episode.json"

    def save():
        trace = {"schema_version": 1, "complete": True, "identity": identity, "calls": []}
        packer = Packer()
        for index, pair in enumerate(packets):
            call = {}
            for kind, value in zip(("request", "response"), pair):
                path = trace_root / f"{index}.{kind}.msgpack"
                path.write_bytes(packer.pack(value))
                call[kind] = {"path": path.name, "sha256": sha256_file(path)}
            trace["calls"].append(call)
        (trace_root / "trace.json").write_text(json.dumps(trace))
        for path, key in ((tmp_path / "episode.json", "result_sha256"),
                          (tmp_path / "episode.npz", "actions_sha256"),
                          (trace_root / "trace.json", "trace_sha256")):
            job[key] = sha256_file(path)
    save()
    return tmp_path, manifest, job, packets, save


def test_valid_episode(episode):
    root, manifest, job, _, _ = episode
    result, _ = verify_episode(root, manifest, job)
    assert result["success"] is False


@pytest.mark.parametrize("corruption", ["seed", "action", "initial", "mid_reset"])
def test_rejects_semantically_invalid_trace_despite_valid_hashes(episode, corruption):
    root, manifest, job, packets, save = episode
    if corruption == "seed":
        packets[0][1]["benchmark_seed"] = 1
    elif corruption == "action":
        packets[1][1]["action"] = np.ones(7, dtype=np.float32)
    elif corruption == "initial":
        packets[1][0]["libero_observation"]["robot0_eef_pos"][0] = 1
    else:
        packets[1][0]["reset"] = True
    save()
    with pytest.raises(ConfigurationError):
        verify_episode(root, manifest, job)


def test_refuses_partial_campaign(tmp_path):
    (tmp_path / "campaign.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ConfigurationError, match="incomplete"):
        compare(tmp_path, tmp_path)
