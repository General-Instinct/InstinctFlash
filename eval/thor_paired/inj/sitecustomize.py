# T3 injection — put this directory FIRST on PYTHONPATH of a lerobot_eval subprocess.
# Same pattern as the LIBERO acceptance run's static-capture hook (e2e_sim/inj, replays=268):
# wrap PI05Policy.from_pretrained so the eval harness never needs a fork.
#
#   IFL_THOR_REMOTE=1   arm B — return a RemoteEnginePolicy proxy (weights never load here;
#                       the Thor engine serves them). IFL_THOR_HOST/IFL_THOR_PORT locate the
#                       server, IFL_ENGINE_SEED seeds the engine's per-episode flow noise.
#   IFL_ACTION_LOG=path both arms — JSONL action-chunk log (the pairing/null-control evidence).
#                       With IFL_THOR_REMOTE unset this wraps the LOCAL policy's
#                       predict_action_chunk/reset so arm A produces the same evidence format.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TP = os.path.dirname(_HERE)                                    # .../eval/thor_paired

if os.environ.get("IFL_THOR_REMOTE") == "1":
    try:
        sys.path.insert(0, _TP)
        import lerobot.policies.pi05.modeling_pi05 as M
        _orig = M.PI05Policy.from_pretrained.__func__

        def _wrapped(cls, *a, **k):
            from remote_policy import RemoteEnginePolicy
            config = k.get("config")
            if config is None:                                  # not the make_policy path
                raise RuntimeError("IFL_THOR_REMOTE=1 requires make_policy-style "
                                   "from_pretrained(config=...) so no weights are loaded")
            proxy = RemoteEnginePolicy(
                config,
                os.environ.get("IFL_THOR_HOST", "100.68.159.80"),
                int(os.environ.get("IFL_THOR_PORT", "29930")),
                log_path=os.environ.get("IFL_ACTION_LOG"),
                engine_seed=int(os.environ.get("IFL_ENGINE_SEED", "7")))
            print(f"[T3] remote engine proxy armed -> ws://{proxy._host}:{proxy._port} "
                  f"(weights NOT loaded locally)", file=sys.stderr, flush=True)
            return proxy

        M.PI05Policy.from_pretrained = classmethod(_wrapped)
        print("[T3] from_pretrained hook armed (remote arm B)", file=sys.stderr, flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"[T3] sitecustomize FAILED: {e!r}", file=sys.stderr, flush=True)

elif os.environ.get("IFL_ACTION_LOG"):
    try:
        import lerobot.policies.pi05.modeling_pi05 as M
        _orig = M.PI05Policy.from_pretrained.__func__

        def _wrapped(cls, *a, **k):
            policy = _orig(cls, *a, **k)
            _install_chunk_logger(policy, os.environ["IFL_ACTION_LOG"])
            return policy

        def _install_chunk_logger(policy, path):
            import hashlib
            import json
            import time
            import torch
            log = open(path, "a", buffering=1)
            state = {"ep": -1, "chunk": 0}
            orig_reset, orig_chunk = policy.reset, policy.predict_action_chunk

            def reset():
                state["ep"] += 1
                state["chunk"] = 0
                log.write(json.dumps({"event": "reset", "ep": state["ep"],
                                      "ts": round(time.time(), 3)}) + "\n")
                return orig_reset()

            def chunk_logged(batch, **kw):
                actions = orig_chunk(batch, **kw)
                n = policy.config.n_action_steps
                a32 = actions[:, :n].detach().to("cpu", torch.float32).numpy()
                task = batch.get("task")
                log.write(json.dumps({
                    "event": "chunk", "ep": state["ep"], "chunk": state["chunk"],
                    "prompt": (task[0] if isinstance(task, (list, tuple)) and task else None),
                    "actions_sha": hashlib.sha256(a32.tobytes()).hexdigest()[:16],
                    "actions": a32[0].tolist(), "ts": round(time.time(), 3)}) + "\n")
                state["chunk"] += 1
                return actions

            def torch_float32():
                import torch
                return torch.float32

            policy.reset = reset
            policy.predict_action_chunk = chunk_logged
            print(f"[T3] local action logger armed -> {path}", file=sys.stderr, flush=True)

        M.PI05Policy.from_pretrained = classmethod(_wrapped)
        print("[T3] from_pretrained hook armed (local arm A, logging)",
              file=sys.stderr, flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"[T3] sitecustomize FAILED: {e!r}", file=sys.stderr, flush=True)
