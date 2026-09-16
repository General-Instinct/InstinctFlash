"""pi05 LIBERO closed-loop + latency driver: the schedule-sweep surface for family #2.

`instinctflash_driver.py` deliberately refuses closed-loop requests; this driver is the LIBERO
simulator half for the pi05 backbone, written against DRIVER_CONTRACT.md for the schedule-sweep
plan type. One client process serves ONE job and exits (the runner's isolation protocol), but a
LIBERO episode job would otherwise pay the full LeRobot build+load floor (~1-2 min measured)
for a ~30 s episode, multiplied by every job in a thousand-episode sweep. So the driver splits
client/server, exactly the split DRIVER_CONTRACT.md names for simulator isolation:

* the CLIENT is the per-job process the runner spawns. It resolves the locked snapshot, then
  forwards the episode request over a per-(GPU, revision) unix socket and writes the result.
* the SERVER is spawned on demand (its own session; it outlives job timeouts), loads the policy
  ONCE, and runs one fully seeded episode per request. Both arms of every pair are served by
  the same loaded weights on the same GPU — the pairing is tighter, not looser, than separate
  processes, and the baseline-repeat noise-floor arm measures whatever residue remains.

Episode protocol (LOCKED, the campaign protocol of research log 2026-08-21; success numbers are
only comparable under all four):

* ``n_action_steps=10`` — the checkpoint ships 50 (whole-chunk open loop) and that costs ~25
  points; the community reproduction used 10.
* ``compile_model=false`` on every arm (the flag's branch also flips process TF32 as a side
  effect, before any wrapper is built).
* one episode per process-request, SyncVectorEnv(n_envs=1) — concurrency blinds EGL rendering
  (the black-screen failure class); solo sync envs are the proven configuration.
* upstream numeric environment: ``lerobot_eval``'s own ``allow_tf32=True`` /
  ``cudnn.benchmark=True`` process state, on both arms. The frontier's latency/Hz column is
  measured under the same flags so it describes the surface the episodes actually ran on.
  MuJoCo physics is CPU-side; the simulator never sees these GPU knobs (contract §TF32).

Schedule dispatch: the arm's ``operating_point.schedule.nfe`` must be exactly ``{"action": N}``
(pi05 declares one stream; a bare or misnamed phase is a refusal, mirroring the sweep spec
validator). The N-step grid is upstream's own ``euler_integrate`` (``t = 1, (N-1)/N, ..., 1/N``);
the stock variant changes only ``num_inference_steps``. The explicitly named
``instinctflash_capture`` comparison keeps NFE=10 and installs the adapter's
loop-constant hoists and static-KV graph on the same verified LeRobot policy.
Separate sockets isolate stock/capture model objects. Numerical settings and
processors remain identical; this measures adapter passes, not the full Runtime
facade and its separate precision lease.

Seed discipline: LIBERO init states are fixed files indexed by ``init_state_id``, which upstream
ties to the (episode_index, n_envs) construction — NOT to the reset seed. A per-episode driver
must therefore map the request seed onto the init-state index itself: ``init_state_id =
requested_seed % n_init_states`` (the wrapper applies the modulo), identical across arms by
construction, and distinct seeds within a suite map to distinct init states because the
registry's LIBERO seed bases are multiples of 50. The same requested seed also seeds torch/np
(policy noise: flow matching x_1) and the env reset.

Latency runs in-process (no server): ``Pi05Stock``'s serve path with the arm's
``num_inference_steps`` and the eval numeric flags, timed by the shared ``run_latency``.

Action canonicalization (part of this driver's revision): closed-loop digests hash the float64
C-order flattening of the full applied-action trace ``(T, action_dim)``, the transport actions
after the env postprocessor — via the same packed-double SHA-256 as ``instinctflash_driver``.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vla.instinctflash_driver import (  # noqa: E402
    DriverRefusal,
    Pi05Stock,
    action_digest,
    driver_revision,
    resolve_snapshot,
    run_latency,
    seed_everything,
)
from benchmarks.vla.util import load_json, sha256_json, write_json_atomic  # noqa: E402

#: The locked closed-loop protocol knob (see module docstring). Not configurable per arm:
#: absolute success moves with it, so it is identical for every arm of every sweep.
N_ACTION_STEPS = 10

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: How long a client waits for the server to come up (first request pays model load).
SERVER_CONNECT_DEADLINE_S = 900.0
#: A server with no connections for this long exits; campaigns leave no daemons behind.
SERVER_IDLE_EXIT_S = 3600.0


# ----------------------------------------------------------------------------------------------
# request interpretation (pure; unit-tested without torch)


def schedule_steps(request: dict) -> int:
    """The arm's declared action NFE, or a refusal — this driver serves schedule sweeps."""
    schedule = request["arm"]["operating_point"].get("schedule")
    if not isinstance(schedule, dict) or not isinstance(schedule.get("nfe"), dict):
        raise DriverRefusal(
            "arm declares no operating_point.schedule.nfe: this driver serves schedule-sweep "
            "arms (benchmarks.vla.schedule_sweep); a fixed-schedule pi05 arm belongs to "
            "instinctflash_driver.py"
        )
    nfe = schedule["nfe"]
    if set(nfe) != {"action"}:
        raise DriverRefusal(
            f"pi05 declares exactly one stream ('action'); got nfe phases {sorted(nfe)!r}. "
            "A misnamed phase would silently serve the baseline schedule."
        )
    steps = nfe["action"]
    if not isinstance(steps, int) or steps < 1:
        raise DriverRefusal(f"nfe['action'] must be a positive integer, got {steps!r}")
    guidance = schedule.get("guidance")
    if guidance is not None:
        # pi05 has no classifier-free guidance: the guidance leg of its operating point is fixed
        # at none/1.0, so a sweep point may state that (and get batch-1 forwards reported) but a
        # point requesting a negative branch cannot be served and must not be silently run as
        # the baseline combine.
        from instinctflash.descriptors.guidance import GuidanceDeclarationError, resolve

        if not isinstance(guidance, dict) or set(guidance) - {"action"}:
            raise DriverRefusal(
                f"pi05 declares exactly one stream ('action'); guidance names {sorted(guidance)!r}"
                if isinstance(guidance, dict) else "schedule.guidance must be an object"
            )
        try:
            resolved = resolve(guidance, {"action": ("none", 1.0)})
        except GuidanceDeclarationError as error:
            raise DriverRefusal(str(error)) from None
        if resolved["action"].negative_branch:
            raise DriverRefusal(
                f"pi05 has no classifier-free guidance; the point requests "
                f"{resolved['action'].describe()} (a negative branch), which this driver cannot serve"
            )
    return steps


def optimization_variant(request: dict) -> str:
    """Named implementation change, independent of the action schedule/numeric flags."""
    variant = request["arm"]["operating_point"].get("optimization", "stock")
    if variant not in {"stock", "instinctflash_capture"}:
        raise DriverRefusal(f"unknown pi05 optimization: {variant!r}")
    if variant == "instinctflash_capture" and schedule_steps(request) != 10:
        raise DriverRefusal("capture comparison preserves original NFE=10")
    return variant


def parse_closed_loop_task(request: dict) -> tuple[str, int]:
    """'libero_spatial/3' -> ('libero_spatial', 3), cross-checked against the suite id."""
    task = str(request["task"])
    suite_name, _, task_id = task.rpartition("/")
    if suite_name not in LIBERO_SUITES or not task_id.isdigit():
        raise DriverRefusal(f"not a LIBERO closed-loop task: {task!r}")
    if request["suite"]["id"] != suite_name:
        raise DriverRefusal(
            f"task {task!r} does not belong to suite {request['suite']['id']!r}"
        )
    return suite_name, int(task_id)


def episode_request(request: dict, snapshot: Path) -> dict:
    """The server-side episode order for one job, fully determined by the plan request."""
    if request["suite"]["seed_strategy"] != "fixed":
        raise DriverRefusal(
            "LIBERO suites preregister fixed seeds (init states are fixed files); "
            f"got {request['suite']['seed_strategy']!r}"
        )
    suite_name, task_id = parse_closed_loop_task(request)
    variant = optimization_variant(request)
    return {
        **({"optimization": variant} if "optimization" in request["arm"]["operating_point"] else {}),
        "op": "episode",
        "suite": suite_name,
        "task_id": task_id,
        "seed": int(request["requested_seed"]),
        "num_inference_steps": schedule_steps(request),
        "n_action_steps": N_ACTION_STEPS,
        "snapshot": str(snapshot),
        "model_revision": request["model"]["checkpoint"]["revision"],
    }


def socket_path(revision: str, optimization: str = "stock") -> Path:
    """One server per (visible GPU set, checkpoint revision, protocol constant)."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "_") or "none"
    suffix = "" if optimization == "stock" else "-capture"
    return Path("/tmp") / f"ifl-pi05-libero-{revision[:12]}-{driver_revision()[:12]}-nas{N_ACTION_STEPS}-gpu{gpu}{suffix}.sock"


# ----------------------------------------------------------------------------------------------
# client


def _spawn_server(sock: Path, snapshot: Path, revision: str, optimization: str = "stock") -> None:
    log_path = sock.with_suffix(".log")
    with log_path.open("a") as log:
        subprocess.Popen(
            [
                sys.executable, os.path.abspath(__file__), "--serve",
                "--socket", str(sock), "--snapshot", str(snapshot), "--revision", revision,
                "--optimization", optimization,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # a job timeout kills the client's group, never the server
        )


def _call_server(sock: Path, payload: dict, snapshot: Path, revision: str) -> dict:
    """Send one request, spawning the server if none is listening. Retries until deadline."""
    deadline = time.monotonic() + SERVER_CONNECT_DEADLINE_S
    spawned = False
    while True:
        try:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(str(sock))
            break
        except OSError:
            connection.close()
            if not spawned:
                _spawn_server(sock, snapshot, revision, payload.get("optimization", "stock"))
                spawned = True
            if time.monotonic() > deadline:
                raise DriverRefusal(
                    f"no LIBERO policy server on {sock} within {SERVER_CONNECT_DEADLINE_S:.0f}s "
                    f"(see {sock.with_suffix('.log')})"
                )
            time.sleep(2.0)
    try:
        connection.sendall(json.dumps(payload).encode() + b"\n")
        connection.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            data = connection.recv(1 << 16)
            if not data:
                break
            chunks.append(data)
    finally:
        connection.close()
    if not chunks:
        raise DriverRefusal(f"LIBERO policy server on {sock} closed without replying")
    reply = json.loads(b"".join(chunks).decode())
    if reply.get("error"):
        raise DriverRefusal(f"server refused episode: {reply['error']}")
    return reply


def run_closed_loop(job: dict) -> dict:
    request = job["request"]
    from benchmarks.vla.adapters import validate_bound_adapter
    validate_bound_adapter(request)
    from benchmarks.vla.pi05_scenes import load_scene
    scene, sources, manifest_hash = load_scene(request)
    checkpoint = request["model"]["checkpoint"]
    snapshot = resolve_snapshot(checkpoint["id"], checkpoint["revision"])
    order = episode_request(request, snapshot)
    order.update(scene=scene, scene_sources=sources)
    sock = socket_path(checkpoint["revision"], optimization_variant(request))
    reply = _call_server(sock, order, snapshot, checkpoint["revision"])
    if reply["model_revision"] != checkpoint["revision"]:
        raise DriverRefusal(
            f"server loaded revision {reply['model_revision']}, plan locked "
            f"{checkpoint['revision']}"
        )
    if reply.get("optimization", "stock") != optimization_variant(request):
        raise DriverRefusal("server optimization differs from plan")
    evidence = reply.get("optimization_evidence", {})
    if optimization_variant(request) == "instinctflash_capture":
        if (evidence.get("variant") != "instinctflash_capture"
                or "graph_capture_static_kv" not in evidence.get("applied", [])
                or not evidence.get("self_checks") or any(v is None for v in evidence["self_checks"])):
            raise DriverRefusal("missing capture execution/self-check evidence")
    return {
        "metrics": {
            "success": bool(reply["success"]),
            "action_digest": reply["action_digest"],
            "finite": bool(reply["finite"]),
            "executed_steps": int(reply["n_env_steps"]),
            "action_values": reply["action_values"],
        },
        "environment_fingerprint": reply["environment_fingerprint"],
        "scene_provenance": {"optimization_evidence": reply.get("optimization_evidence", {"variant": "stock"}),
            "scene": scene, "scene_sha256": sha256_json(scene),
            "scene_manifest_sha256": manifest_hash, "evaluation_mode": "paused_simulation"},
    }


# ----------------------------------------------------------------------------------------------
# latency arm (in-process; the sweep's per-point cycle cost on the eval surface)


def _apply_eval_numeric_environment() -> None:
    """lerobot_eval's own process flags, applied identically on every arm (module docstring)."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class Pi05ScheduleArm(Pi05Stock):
    """The stock serve path at the arm's declared step schedule, under the eval numeric flags."""

    def __init__(self, request: dict):
        super().__init__(request)
        _apply_eval_numeric_environment()
        self._policy.config.num_inference_steps = schedule_steps(request)


def _fingerprint_facts() -> dict:
    import lerobot
    import torch

    return {
        "driver": "pi05_libero_driver",
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "lerobot": lerobot.__version__,
        "protocol": {
            "n_action_steps": N_ACTION_STEPS,
            "allow_tf32": True,
            "cudnn_benchmark": True,
            "compile_model": False,
        },
    }


def run_job(job: dict) -> dict:
    request = job["request"]
    if sha256_json(request) != job["request_sha256"]:
        raise DriverRefusal("request digest mismatch")
    if request["schema_version"] != 1:
        raise DriverRefusal(f"unsupported request schema {request['schema_version']!r}")
    if request["model"]["backbone"] != "pi05":
        raise DriverRefusal("this driver serves the pi05 backbone on LIBERO only")
    revision = driver_revision()
    if revision.endswith("-dirty"):
        raise DriverRefusal(
            f"driver checkout is dirty ({revision}); benchmark evidence must come from a "
            f"committed revision"
        )
    if revision != job["driver"]["revision"]:
        raise DriverRefusal(
            f"driver revision {revision} does not match the planned {job['driver']['revision']}"
        )
    kind = request["suite"]["kind"]
    scene_provenance = {}
    if kind == "closed_loop":
        outcome = run_closed_loop(job)
        metrics = outcome["metrics"]
        scene_provenance = outcome["scene_provenance"]
        fingerprint = outcome["environment_fingerprint"]
    elif kind == "latency":
        if optimization_variant(request) != "stock":
            raise DriverRefusal("capture evaluation currently supports closed-loop jobs only")
        if request["suite"]["seed_strategy"] != "fixed":
            raise DriverRefusal("latency suites preregister fixed seeds")
        arm = Pi05ScheduleArm(request)
        try:
            metrics = run_latency(arm, request, int(request["requested_seed"]))
        finally:
            arm.close()
        fingerprint = sha256_json({**_fingerprint_facts(), "arm": "in_process_latency"})
    else:
        raise DriverRefusal(
            f"suite kind {kind!r} is not served: contract/open-loop pi05 evidence belongs to "
            f"instinctflash_driver.py"
        )
    return {
        "schema_version": 1,
        "job_id": job["job_id"],
        "request_sha256": job["request_sha256"],
        "status": "completed",
        "resolved_seed": int(request["requested_seed"]),
        "metrics": metrics,
        "provenance": {
            "model_revision": request["model"]["checkpoint"]["revision"],
            "driver_revision": revision,
            "environment_fingerprint": fingerprint,
            "synthetic": False,
            **scene_provenance,
        },
    }


# ----------------------------------------------------------------------------------------------
# server


class _PolicyServer:
    """Loads the policy once; runs one fully seeded LIBERO episode per connection."""

    def __init__(self, snapshot: Path, revision: str, optimization: str = "stock"):
        if optimization not in {"stock", "instinctflash_capture"}:
            raise DriverRefusal("unsupported server optimization")
        self._optimization = optimization
        self._capture_runtime = None
        self._applied = []
        _apply_eval_numeric_environment()
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env_config

        self._snapshot = snapshot
        self._revision = revision
        env_cfg = make_env_config("libero", task=LIBERO_SUITES[0], task_ids=[0])
        policy_cfg = PreTrainedConfig.from_pretrained(snapshot)
        policy_cfg.pretrained_path = str(snapshot)
        policy_cfg.compile_model = False
        policy_cfg.n_action_steps = N_ACTION_STEPS
        self._policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        from benchmarks.vla.pi05_weights import verify_loaded_weights
        self._weight_identity = verify_loaded_weights(self._policy, snapshot)
        print("verified checkpoint tensors:", self._weight_identity, flush=True)
        self._policy.eval()
        self._policy_cfg = policy_cfg
        device = str(policy_cfg.device)
        self._pre, self._post = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(snapshot),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        if optimization == "instinctflash_capture":
            from instinctflash import Runtime
            self._capture_runtime = Runtime.from_pretrained(
                "lerobot/pi05_libero_finetuned_v044", revision=revision, device=device,
                tier_ceiling="bitexact")
            from pi05_iwm.adapter import Pi05Adapter
            self._applied = Pi05Adapter.install(self._policy, self._capture_runtime.plan, device=device)
            if "graph_capture_static_kv" not in self._applied:
                raise DriverRefusal("requested capture was not installed")
        self._fingerprint = sha256_json({**_fingerprint_facts(), "arm": "episode_server",
                                       "optimization": optimization, "applied": self._applied,
                                       "loaded_weights": self._weight_identity})
        self.episodes = 0
        #: set when the EGL context is caught dead — the server must exit so the next client
        #: gets a fresh renderer instead of a blind policy scoring honest-looking zeros
        self.poisoned = False

    def episode(self, order: dict) -> dict:
        import numpy as np
        import torch
        from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
        from lerobot.scripts.lerobot_eval import rollout

        if order.get("optimization", "stock") != self._optimization:
            return {"error": "server optimization mismatch"}
        if self._optimization == "instinctflash_capture" and order["num_inference_steps"] != 10:
            return {"error": "capture comparison preserves original NFE=10"}
        if order["snapshot"] != str(self._snapshot):
            return {"error": f"server holds {self._snapshot}, request wants {order['snapshot']}"}
        if order["n_action_steps"] != N_ACTION_STEPS:
            return {"error": f"protocol is n_action_steps={N_ACTION_STEPS}, immutable per arm"}
        from benchmarks.vla.pi05_scenes import identity, checked_reset
        if order.get("scene_sources") != identity():
            return {"error": "scene source/environment changed"}
        seed = int(order["seed"])
        self._policy.config.num_inference_steps = int(order["num_inference_steps"])
        env_cfg = make_env_config(
            "libero", task=order["suite"], task_ids=[int(order["task_id"])]
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self._policy_cfg
        )
        seed_everything(seed)
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        vec = envs[order["suite"]][int(order["task_id"])]
        try:
            # init state = seed mod n_init_states, identically on both arms (module docstring);
            # upstream ties it to episode_index, which is always 0 for a 1-env vec. `.unwrapped`
            # reaches the LiberoEnv through any wrapper (gym wrappers do not forward attribute
            # WRITES).
            vec.envs[0].unwrapped.init_state_id = seed
            with checked_reset(vec, order["scene"]), torch.no_grad():
                data = rollout(
                    vec,
                    self._policy,
                    env_preprocessor,
                    env_postprocessor,
                    self._pre,
                    self._post,
                    seeds=[seed],
                )
            # The black-screen class (lingbot-va eval traps; measured on this harness's own
            # ancestors): a dying EGL context renders BLACK, the blind policy scores an
            # honest-looking failure, and the number poisons the paired comparison. Probe the
            # same renderer the observations came from; near-zero luminance invalidates the
            # episode and poisons the server so the retry gets a fresh process.
            try:
                frame = vec.envs[0].unwrapped.render()
            except Exception:  # noqa: BLE001 - detector failure must not fail the episode
                frame = None
            if frame is not None and float(np.asarray(frame).mean()) <= 3.0:
                self.poisoned = True
                return {
                    "error": "blackscreen: near-zero luminance render, EGL context is dead; "
                    "server exits so the retried job gets a fresh renderer"
                }
        finally:
            vec.close()
        actions = np.asarray(data["action"][0].numpy(), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
            return {"error": "invalid executed action trace"}
        self.episodes += 1
        return {
            "success": bool(data["success"][0].any().item()),
            "action_digest": action_digest(actions.ravel()),
            "finite": bool(np.isfinite(actions).all()),
            "n_env_steps": int(actions.shape[0]),
            "action_values": actions.ravel().tolist(),
            "model_revision": self._revision,
            "environment_fingerprint": self._fingerprint,
            "episodes_served": self.episodes,
            "optimization": self._optimization,
            "optimization_evidence": {
                "variant": self._optimization, "applied": self._applied,
                "scope": "InstinctFlash adapter passes on the same LeRobot evaluation policy; not the full Runtime facade/precision lease",
                "plan": self._capture_runtime.plan.explain() if self._capture_runtime else None,
                "self_checks": [r.params.get("self_check") for r in self._capture_runtime.plan.results
                                if r.name == "graph_capture"] if self._capture_runtime else [],
                "loaded_weights": self._weight_identity,
            },
        }


def serve(sock_path: Path, snapshot: Path, revision: str, optimization: str = "stock") -> int:
    lock_stream = sock_path.with_suffix(".flock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another server owns {sock_path}; exiting", flush=True)
        return 0
    with contextlib.suppress(FileNotFoundError):
        sock_path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(8)
    listener.settimeout(SERVER_IDLE_EXIT_S)
    print(f"loading policy from {snapshot}", flush=True)
    server = _PolicyServer(snapshot, revision, optimization)
    print(f"serving on {sock_path}", flush=True)
    try:
        while True:
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                print(f"idle {SERVER_IDLE_EXIT_S:.0f}s; exiting", flush=True)
                return 0
            with contextlib.closing(connection):
                data = b""
                while not data.endswith(b"\n"):
                    chunk = connection.recv(1 << 16)
                    if not chunk:
                        break
                    data += chunk
                if not data.strip():
                    continue
                order = json.loads(data.decode())
                if order.get("op") == "ping":
                    reply = {"ok": True, "episodes_served": server.episodes}
                else:
                    started = time.monotonic()
                    try:
                        reply = server.episode(order)
                    except Exception as error:  # noqa: BLE001 - reported to the client, loudly
                        reply = {"error": f"{type(error).__name__}: {error}"}
                    reply["episode_wall_s"] = round(time.monotonic() - started, 3)
                    where = (
                        f"episode {order.get('suite')}/{order.get('task_id')} seed "
                        f"{order.get('seed')} nfe {order.get('num_inference_steps')}"
                    )
                    if "error" in reply:
                        print(f"{where}: ERROR {reply['error']}", flush=True)
                    else:
                        summary = {
                            key: reply[key]
                            for key in ("success", "n_env_steps", "episode_wall_s")
                            if key in reply
                        }
                        print(f"{where}: {json.dumps(summary)}", flush=True)
                connection.sendall(json.dumps(reply).encode())
            if server.poisoned:
                print("renderer poisoned; exiting for a fresh process", flush=True)
                return 1
    finally:
        listener.close()
        if server._capture_runtime is not None:
            server._capture_runtime.close()
        with contextlib.suppress(FileNotFoundError):
            sock_path.unlink()


# ----------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-config", type=Path,
                        help="Run the source-locked FlashRT Native/FP8 SM120 paired campaign")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--revision", type=str)
    parser.add_argument("--optimization", choices=["stock", "instinctflash_capture"], default="stock")
    args = parser.parse_args()
    if args.matched_config:
        from benchmarks.vla.pi05_sm120_libero import campaign
        return campaign(args.matched_config)
    if args.serve:
        if not (args.socket and args.snapshot and args.revision):
            parser.error("--serve requires --socket, --snapshot and --revision")
        return serve(args.socket, args.snapshot, args.revision, args.optimization)
    if not (args.request and args.output):
        parser.error("--request and --output are required")
    job = load_json(args.request)
    result = run_job(job)
    write_json_atomic(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
