"""One frozen RoboTwin episode -> one standard benchmark result.

The environment/configuration and rollout remain the pinned upstream LingBot client.
This wrapper freezes scenes BEFORE either arm runs, intercepts actions at take_action,
and supplies a bounded, identity-checked remote policy. No simulator imports at analysis
time. Only wan_va/raw-wire is implemented; other model bridges must be explicit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.vla.remote_policy import RemotePolicy
from benchmarks.vla.result import validate_result
from benchmarks.vla.util import ConfigurationError, load_json, sha256_file, sha256_json, write_json_atomic

PROTOCOL = "wan-va-robotwin-paused-v1"
RESET_POLICY = "fresh-clutter-bounds-v1"
SETTINGS = {"robotwin50_easy": "demo_clean", "robotwin50_hard": "demo_randomized"}
CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist",
           "observation.images.cam_right_wrist")


def driver_revision():
    return "robotwin-v1:" + sha256_json({p.name: sha256_file(p) for p in (
        Path(__file__), Path(__file__).with_name("remote_policy.py"),
        Path(__file__).with_name("adapters.py"), Path(__file__).parent / "config/adapters.json")})


def validate_request(job):
    request = job["request"]
    if sha256_json(request) != job["request_sha256"]:
        raise ConfigurationError("request digest mismatch")
    if job["driver"]["revision"] != driver_revision():
        raise ConfigurationError("robotwin driver content revision differs from plan; rebuild plan")
    suite = request["suite"]
    if suite["kind"] != "closed_loop" or suite["id"] not in SETTINGS:
        raise ConfigurationError("robotwin driver serves only RoboTwin closed_loop suites")
    if request["model"]["backbone"] != "wan_va":
        raise ConfigurationError("no RoboTwin policy bridge for this backbone; implemented: wan_va")
    if suite["protocol"].get("setting") != SETTINGS[suite["id"]]:
        raise ConfigurationError("RoboTwin suite setting mismatch")
    if suite["protocol"].get("bridge") != PROTOCOL or suite["protocol"].get("evaluation_mode") != "paused_simulation":
        raise ConfigurationError("plan must explicitly declare the paused RoboTwin bridge protocol")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", request["task"]):
        raise ConfigurationError("invalid RoboTwin task name")
    if type(request["requested_seed"]) is not int or request["requested_seed"] < 0:
        raise ConfigurationError("requested_seed must be a non-negative integer")
    if suite["seed_strategy"] not in {"fixed", "increment_until_stable"}:
        raise ConfigurationError("unsupported seed strategy")
    if type(suite["seed_max_attempts"]) is not int or suite["seed_max_attempts"] < 1:
        raise ConfigurationError("seed_max_attempts must be positive")
    from benchmarks.vla.adapters import validate_bound_adapter
    validate_bound_adapter(request)
    return request


def scene_key(request):
    return f"{request['suite']['id']}/{request['task']}/{request['requested_seed']}"


def source_identity(robotwin: Path, lingbot: Path):
    """Bind actual local source/config bytes, including existing local protocol patches.

    Assets are bound by the dataset revision and explicit asset tree content hash below;
    the large assets hash is prepared once and must be reverified before a new campaign.
    """
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=robotwin, text=True).strip()
    files = {}
    for label, root, directories in (
        ("robotwin", robotwin, ("envs", "task_config", "description")),
        ("lingbot", lingbot, ("evaluation/robotwin",)),
    ):
        for directory in directories:
            base = root / directory
            if not base.is_dir():
                raise ConfigurationError(f"missing upstream directory {base}")
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix in {".py", ".yml", ".yaml", ".json"}:
                    files[f"{label}/{p.relative_to(root)}"] = sha256_file(p)
    return {"robotwin_revision": revision, "sources_sha256": sha256_json(files)}


def assets_identity(robotwin: Path):
    assets = robotwin / "assets"
    if not assets.is_dir():
        raise ConfigurationError("RoboTwin assets directory is missing")
    # Hash bytes, not timestamps: a changed mesh can change success without changing code.
    files = {str(p.relative_to(assets)): sha256_file(p) for p in sorted(assets.rglob("*"))
             if p.is_file()}
    if not files:
        raise ConfigurationError("RoboTwin assets directory is empty")
    return sha256_json(files)


@contextlib.contextmanager
def patched(obj, **values):
    old = {key: getattr(obj, key) for key in values}
    for key, value in values.items():
        setattr(obj, key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(obj, key, value)


@contextlib.contextmanager
def environment(**values):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def import_client(robotwin, lingbot):
    os.environ["ROBOTWIN_ROOT"] = str(robotwin)
    sys.path.insert(0, str(lingbot))
    # Upstream imports the simulator and changes cwd. The CLI runs in an isolated job process.
    return importlib.import_module("evaluation.robotwin.eval_polict_client_openpi")


@contextlib.contextmanager
def fresh_clutter_bounds(env):
    """Keep upstream table offsets from accumulating in mutable default arguments."""
    original = getattr(env, "get_cluttered_table", None)
    if original is None:
        yield
        return
    signature = inspect.signature(original)
    def isolated(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name in ("xlim", "ylim", "zlim"):
            if name in bound.arguments:
                bound.arguments[name] = list(bound.arguments[name])
        return original(*bound.args, **bound.kwargs)
    with patched(env, get_cluttered_table=isolated):
        yield


def with_upstream_setup(client, request, callback, work):
    """Reuse upstream main's robot/camera/task configuration without its CLI or socket retry."""
    captured = []
    def evaluate(task, env, args, model, start, **kwargs):
        try:
            with fresh_clutter_bounds(env):
                captured.append(callback(env, args, kwargs))
            return request["requested_seed"], int(bool(captured[-1].get("success", False)))
        finally:
            env.close_env()
    with patched(client, eval_policy=evaluate, WebsocketClientPolicy=lambda **kw: None):
        client.main({"task_name": request["task"], "task_config": SETTINGS[request["suite"]["id"]],
                     "ckpt_setting": "instinctflash-benchmark", "save_root": str(work),
                     "policy_name": "InstinctFlash", "video_guidance_scale": 5.0,
                     "action_guidance_scale": 1.0, "seed": 0, "test_num": 1, "port": 0})
    if len(captured) != 1:
        raise ConfigurationError("upstream main did not execute exactly one evaluation callback")
    return captured[0]


def prepare_scene(client, request, work, excluded=()):
    """Expert-only scene discovery; no model server is connected in this phase."""
    def prepare(env, args, kwargs):
        import numpy as np
        settings = dict(args, eval_mode=True, render_freq=0, eval_video_log=False)
        settings.pop("eval_video_save_dir", None)
        first = request["requested_seed"]
        attempts = 1 if request["suite"]["seed_strategy"] == "fixed" else request["suite"]["seed_max_attempts"]
        for seed in range(first, first + attempts):
            if seed in excluded:
                continue
            try:
                env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **settings)
                episode = env.play_once()
                valid = env.plan_success and env.check_success()
            except client.UnStableError:
                valid = False
            except AssertionError as error:
                # Native grasp planning can return no reachable pose. The upstream
                # expert gate retries that seed; keep this narrow so unrelated bugs
                # still abort preparation rather than becoming selected-away failures.
                if str(error) != "target_pose cannot be None for move action.":
                    raise
                valid = False
            finally:
                env.close_env()
            # Unexpected setup/physics exceptions propagate, not "unsuitable seed".
            if not valid:
                continue
            try:
                env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **settings)
                descriptions = client.generate_episode_descriptions(request["task"], [episode["info"]], 1)
                prompt = str(np.random.choice(descriptions[0]["seen"]))
                initial = initial_state(env.get_obs())
            finally:
                env.close_env()
            return {"requested_seed": first, "resolved_seed": seed, "prompt": prompt,
                    "initial_state_sha256": initial, "task": request["task"],
                    "suite_id": request["suite"]["id"]}
        raise ConfigurationError(f"expert found no valid scene in {attempts} attempts for {scene_key(request)}")
    return with_upstream_setup(client, request, prepare, work)


def initial_state(observation):
    import numpy as np
    values = list(np.asarray(observation["joint_action"]["vector"]).ravel())
    for name in ("left_endpose", "right_endpose", "left_gripper", "right_gripper"):
        values.extend(np.asarray(observation["endpose"][name]).ravel())
    digest = hashlib.sha256(b"".join(struct.pack("!d", float(x)) for x in values))
    for name in ("head_camera", "left_camera", "right_camera"):
        image = np.asarray(observation["observation"][name]["rgb"])
        digest.update(str((image.shape, image.dtype.str)).encode())
        digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


class WanVaWireBridge:
    """The raw upstream reset -> infer -> commit protocol (not Runtime's folded predict)."""
    def __init__(self, remote, scene):
        self.remote, self.scene = remote, scene
        self.phase, self.cycles = "reset", 0

    def infer(self, obs):
        if obs.get("reset"):
            if self.phase != "reset" or obs.get("prompt") != self.scene["prompt"]:
                raise ConfigurationError("unexpected episode reset or prompt")
            result = self.remote.reset_episode(self.scene["prompt"], self.scene["resolved_seed"])
            self.phase = "infer"
            return result
        if obs.get("compute_kv_cache"):
            if self.phase != "commit":
                raise ConfigurationError("commit arrived without an inference")
            frames = obs.get("obs")
            expected = 4 if self.cycles == 0 else 8
            if not isinstance(frames, list) or len(frames) != expected:
                raise ConfigurationError(f"wan_va commit requires {expected} observed frames")
            self.phase = "infer"
            self.cycles += 1
        else:
            if self.phase != "infer":
                raise ConfigurationError("infer arrived before reset/commit")
            self.phase = "commit"
        # Guidance is fixed by the server identity; upstream client's guidance fields are ignored upstream.
        request = {k: v for k, v in obs.items() if k not in {"video_guidance_scale", "action_guidance_scale"}}
        request["save_visualization"] = False
        response = self.remote.infer(request)
        if self.phase == "commit":
            import numpy as np
            action = np.asarray(response.get("action"))
            if action.shape != (16, 2, 16) or not np.isfinite(action).all():
                raise ConfigurationError("wan_va requires finite action shape (16, 2, 16)")
        return response


def rollout(client, request, scene, remote, work, *, bridge_factory=WanVaWireBridge):
    def evaluate(env, args, kwargs):
        import numpy as np
        digest, actions = hashlib.sha256(), []
        setup, take_action = env.setup_demo, env.take_action
        def checked_setup(*a, **kw):
            if kw.get("seed") != scene["resolved_seed"]:
                raise ConfigurationError("upstream substituted a frozen scene")
            result = setup(*a, **kw)
            if initial_state(env.get_obs()) != scene["initial_state_sha256"]:
                raise ConfigurationError("initial robot state differs from frozen scene")
            return result
        def record(action, *a, **kw):
            values = np.asarray(action, dtype=np.float64).ravel()
            if values.size != 16 or not np.isfinite(values).all():
                raise ConfigurationError("invalid controller action")
            before = env.take_action_cnt
            result = take_action(action, *a, **kw)
            # Upstream silently ignores actions after success/step limit. Hash only accepted ones.
            if env.take_action_cnt > before:
                digest.update(b"".join(struct.pack("!d", float(v)) for v in values))
                actions.append(values.tolist())
            return result
        bridge = bridge_factory(remote, scene)
        seeds = Path(work) / "seeds"
        write_json_atomic(seeds / f"{request['task']}.json", {"seeds": [scene["resolved_seed"]]})
        settings = dict(args, eval_video_log=False)
        settings.pop("eval_video_save_dir", None)
        # Re-run the expert gate on the SAME pinned seed. A rejection invalidates the job;
        # the one-entry cache makes upstream exhaustion fatal instead of substituting a scene.
        with patched(env, setup_demo=checked_setup, take_action=record), patched(
            client, generate_episode_descriptions=lambda *a, **kw: [{"seen": [scene["prompt"]]}],
            save_comparison_video=lambda **kw: None,
        ), environment(IWM_SEED_CACHE=seeds, IWM_ACTION_LOG=None):
            _, successes = original_eval[0](request["task"], env, settings, bridge,
                                           scene["resolved_seed"], test_num=1,
                                           instruction_type="seen", save_visualization=False)
        if env.test_num != 1 or not actions or bridge.phase != "infer" or bridge.cycles < 1:
            raise ConfigurationError("incomplete episode or missing final KV commit")
        write_json_atomic(Path(work) / "executed_actions.json", actions)
        return {"finite": True, "success": bool(successes), "action_digest": digest.hexdigest(),
                "executed_steps": len(actions), "cycles": bridge.cycles}
    original_eval = [client.eval_policy]
    return with_upstream_setup(client, request, evaluate, work)


def load_scene(request, reference, protocol=PROTOCOL):
    path = Path(reference["path"]).expanduser()
    if sha256_file(path) != reference["sha256"]:
        raise ConfigurationError("scene manifest digest mismatch")
    manifest = load_json(path)
    if manifest.get("protocol") != protocol:
        raise ConfigurationError("unsupported scene protocol")
    if manifest.get("reset_policy") != RESET_POLICY:
        raise ConfigurationError("reprepare scenes with fresh clutter bounds; legacy reset state is not reproducible")
    identities = [(v["suite_id"], v["task"], v["resolved_seed"]) for v in manifest["scenes"].values()]
    if len(identities) != len(set(identities)):
        raise ConfigurationError("scene manifest reuses a resolved scene as independent evidence")
    scene = manifest["scenes"].get(scene_key(request))
    if not scene or not isinstance(scene.get("prompt"), str) or not scene["prompt"]:
        raise ConfigurationError("missing frozen scene/prompt")
    seed = scene["resolved_seed"]
    first = request["requested_seed"]
    if type(seed) is not int or not first <= seed < first + request["suite"]["seed_max_attempts"]:
        raise ConfigurationError("frozen seed outside planned search range")
    if request["suite"]["seed_strategy"] == "fixed" and seed != first:
        raise ConfigurationError("fixed seed changed")
    if (scene["requested_seed"], scene["task"], scene["suite_id"]) != (first, request["task"], request["suite"]["id"]):
        raise ConfigurationError("scene identity mismatch")
    return manifest, scene


def validate_remote(request, remote):
    identity = remote["identity"]
    checkpoint = request["model"]["checkpoint"]
    if identity.get("model_id") != checkpoint["id"] or identity.get("model_revision") != checkpoint["revision"]:
        raise ConfigurationError("remote checkpoint identity differs from plan")
    if identity.get("protocol") != PROTOCOL or identity.get("seed_mode") != "episode_plus_frame":
        raise ConfigurationError("remote must support wan_va raw wire and per-episode noise seeding")
    if identity.get("synthetic") is not False:
        raise ConfigurationError("a real driver cannot attest a synthetic server")
    if identity.get("execution") != request["arm"]["operating_point"].get("execution"):
        raise ConfigurationError("remote execution differs from planned operating point")
    return identity


def run_job(job, robotwin, lingbot, work):
    request = validate_request(job)
    point = request["arm"]["operating_point"]
    if not point.get("scene_manifest") or not point.get("remote"):
        raise ConfigurationError("draft plan is not executable; bind frozen scenes and a remote identity first")
    manifest, scene = load_scene(request, point["scene_manifest"])
    sources = source_identity(robotwin, lingbot)
    if sources != manifest["sources"] or sources["robotwin_revision"] != request["dataset"]["revision"]:
        raise ConfigurationError("simulator revision/source differs from frozen campaign")
    if assets_identity(robotwin) != manifest["assets_sha256"]:
        raise ConfigurationError("simulator assets differ from frozen campaign")
    identity = validate_remote(request, point["remote"])
    client = import_client(robotwin, lingbot)
    remote = RemotePolicy(point["remote"]["endpoint"], identity,
                          timeout=point["remote"].get("timeout_seconds", 120))
    try:
        metrics = rollout(client, request, scene, remote, work)
    finally:
        remote.close()
    facts = {"python": sys.version, "packages": sorted(
        f"{d.metadata.get('Name')}=={d.version}" for d in importlib.metadata.distributions()),
             "sources": sources, "assets_sha256": manifest["assets_sha256"],
             "remote_identity": identity}
    result = {"schema_version": 1, "job_id": job["job_id"],
              "request_sha256": job["request_sha256"], "status": "completed",
              "resolved_seed": scene["resolved_seed"], "metrics": metrics,
              "provenance": {"model_revision": identity["model_revision"],
                             "driver_revision": driver_revision(), "synthetic": False,
                             "environment_fingerprint": sha256_json(facts),
                             "scene_sha256": sha256_json(scene), "scene": scene,
                             "scene_manifest_sha256": point["scene_manifest"]["sha256"], "protocol": PROTOCOL,
                             "evaluation_mode": "paused_simulation", "facts": facts},
              "diagnostics": {"transport_timings": remote.timings,
                              "timing_claim": "diagnostic roundtrip only; no realtime verdict"}}
    validate_result(result, job)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request", type=Path)
    mode.add_argument("--prepare-plan", type=Path, help="expert-only scene preparation from a draft plan")
    mode.add_argument("--revision", action="store_true")
    mode.add_argument("--make-plan", action="store_true", help="write a quality-only plan and its registry snapshot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robotwin-root", default=os.environ.get("ROBOTWIN_ROOT"))
    parser.add_argument("--lingbot-root", default=os.environ.get("LINGBOT_ROOT"))
    parser.add_argument("--arms", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--profile", choices=("robotwin_quality_smoke", "robotwin_quality"),
                        default="robotwin_quality_smoke")
    parser.add_argument("--margin", type=float, help="explicit acceptable success-rate loss, negative fraction")
    parser.add_argument("--min-pairs", type=int)
    parser.add_argument("--scenes", type=Path, help="bind prepared scenes into every arm; omit for a draft plan")
    args = parser.parse_args(argv)
    if args.revision:
        print(driver_revision())
        return 0
    if args.make_plan:
        if not args.output or not args.arms or args.margin is None or args.min_pairs is None:
            parser.error("--make-plan requires --output, --arms, --margin and --min-pairs")
        make_plan(args.arms, args.output, args.profile, args.margin, args.min_pairs,
                  scenes=args.scenes, registry_path=args.registry)
        return 0
    if not args.output or not args.robotwin_root or not args.lingbot_root:
        parser.error("--output, ROBOTWIN_ROOT and LINGBOT_ROOT are required")
    robotwin, lingbot = Path(args.robotwin_root).resolve(), Path(args.lingbot_root).resolve()
    output = args.output.resolve()
    try:
        if args.prepare_plan:
            from benchmarks.vla.plan import validate_plan
            plan = load_json(args.prepare_plan)
            validate_plan(plan)
            requests = {}
            sources = source_identity(robotwin, lingbot)
            for job in plan["jobs"]:
                request = validate_request(job)
                if request["dataset"]["revision"] != sources["robotwin_revision"]:
                    raise ConfigurationError("local RoboTwin revision differs from plan")
                requests[scene_key(request)] = request
            if output.exists():
                raise ConfigurationError("refusing to replace an existing frozen scene manifest")
            assets = assets_identity(robotwin)
            client = import_client(robotwin, lingbot)
            scenes = {}
            with tempfile.TemporaryDirectory(prefix="ifl-scenes-") as work:
                for key, request in requests.items():
                    used = {v["resolved_seed"] for v in scenes.values()
                            if v["task"] == request["task"] and v["suite_id"] == request["suite"]["id"]}
                    scenes[key] = prepare_scene(client, request, work, used)
            write_json_atomic(output, {"schema_version": 1, "protocol": PROTOCOL,
                                       "reset_policy": RESET_POLICY,
                                       "sources": sources, "assets_sha256": assets, "scenes": scenes})
            print(f"scene manifest sha256: {sha256_file(output)}")
        else:
            # Evidence stays beside the job's pending output; never mix two episodes' traces.
            work = output.parent / (output.stem + "-evidence")
            work.mkdir(parents=True, exist_ok=True)
            result = run_job(load_json(args.request), robotwin, lingbot, work)
            write_json_atomic(output, result)
        return 0
    except (ConfigurationError, OSError, RuntimeError) as error:
        print(f"robotwin driver: {error}", file=sys.stderr)
        return 2


def make_plan(arms_path, output, profile, margin, min_pairs, *, scenes=None, registry_path=None):
    """Explicit quality budget; existing published suite margins remain untouched."""
    import copy
    from benchmarks.vla.plan import build_plan
    from benchmarks.vla.registry import load_registry, Registry, _validate

    if not -1 < margin < 0 or type(min_pairs) is not int or min_pairs < 1:
        raise ConfigurationError("provide an explicit negative quality margin and positive min_pairs")
    output = Path(output).resolve()
    registry_file = output.with_suffix(".registry.json")
    if output.exists() or registry_file.exists():
        raise ConfigurationError("refusing to replace a frozen plan/registry; use a fresh output path")
    raw = copy.deepcopy(load_registry(registry_path).raw)
    for suite in raw["suites"]:
        if suite["id"] in SETTINGS:
            suite["protocol"].update(margin=margin, interval="tango_one_sided95",
                                     evaluation_mode="paused_simulation", bridge=PROTOCOL,
                                     screening=profile.endswith("smoke"))
    raw["profiles"][profile] = {
        "suites": list(SETTINGS), "limits": {"tasks": 1 if profile.endswith("smoke") else None,
        "seeds_per_task": {"closed_loop": 1 if profile.endswith("smoke") else 10}},
        "arm_repeats": 1, "latency": {"warmup": 0, "iterations": 1}}
    arms = load_json(Path(arms_path))
    for arm in arms["arms"]:
        arm["driver"]["revision"] = driver_revision()
        if arm["role"] == "treatment":
            arm["gates"] = {"performance": {"min_speedup": 1.0}, "action": {"mode": "registry"},
                            "success": {"margin": margin, "interval": "tango_one_sided95", "min_pairs": min_pairs}}
        point = arm["operating_point"]
        if scenes:
            point["scene_manifest"] = {"path": str(Path(scenes).resolve()), "sha256": sha256_file(Path(scenes))}
        else:
            point.pop("scene_manifest", None)  # draft plans can prepare scenes but cannot execute
    _validate(raw)
    registry = Registry(raw, sha256_json(raw), registry_file)
    plan = build_plan(registry, arms, profile, ["robbyant/lingbot-va-posttrain-robotwin"])
    if scenes:
        for job in plan["jobs"]:
            request = validate_request(job)
            load_scene(request, request["arm"]["operating_point"]["scene_manifest"])
            validate_remote(request, request["arm"]["operating_point"]["remote"])
    write_json_atomic(registry_file, raw)
    write_json_atomic(output, plan)
    print(f"{'bound' if scenes else 'draft'} quality plan: {output}; registry: {registry_file}")
    return plan


if __name__ == "__main__":
    raise SystemExit(main())
