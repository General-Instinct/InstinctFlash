"""LingBot-VA backend adapter — declarations, plus the thin glue that serves them.

Every field in `lingbot_va_spec()` is a fact read out of the upstream lingbot-va tree (set
`LINGBOT_ROOT`; the historical default is `/home/ubuntu/lingbot-va`), cited to `file:line`.
Nothing there is an optimization; the optimizer derives those. If you find yourself wanting to
add a `use_fast_path=True` field, it belongs in a pass instead.

`LingBotVA` at the bottom is the `BackendAdapter` implementation: declarations, plus install
and serve. It holds no optimization logic either — it maps pass names onto installers and
refuses to serve a plan it could not fully apply.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

from instinctflash.adapters.base import (  # noqa: F401
    ObservationField, ObservationSpec,
    AdapterSpec, CommitMode, GuidanceMode, GuidanceRule, KVLifetime, KVStreamSpec,
    PhaseSpec, PurityKey,
)

# --- geometry, from wan_va/configs/va_robotwin_cfg.py --------------------------------------
_HEIGHT, _WIDTH = 256, 320
_FRAME_CHUNK = 2            # frame_chunk_size
_ACTION_PER_FRAME = 16      # action_per_frame
_PATCH = (1, 2, 2)          # patch_size
# env_type 'robotwin_tshape': cam_high full res, the two wrist cams at half res, composited into
# a T. latent grid = ((256//16)*3)//2 x (320//16) = 24 x 20, patched by (2,2) -> 12 x 10 = 120.
_VIDEO_TOKENS_PER_FRAME = ((_HEIGHT // 16) * 3 // 2) * (_WIDTH // 16) // (_PATCH[1] * _PATCH[2])
_ACTION_TOKENS_PER_FRAME = _ACTION_PER_FRAME


def lingbot_va_spec() -> AdapterSpec:
    return AdapterSpec(
        model_id="lingbot-va-posttrain-robotwin",
        param_bytes=10_179_017_396,  # transformer safetensors, bf16

        # Two CO-EQUAL committed streams. Both are written by the same `update_cache` path
        # (model.py:444-447) and both are persisted with update_cache=2 in _compute_kv_cache
        # (wan_va_server.py:593-601), so both are attended in every later control step. The
        # action stream is NOT scratch — this is the fact that vLLM-Omni's single
        # `tokens_per_frame` + `max_scratch_tokens_per_branch` spec cannot express.
        streams=(
            KVStreamSpec(
                name="video",
                tokens_per_frame=_VIDEO_TOKENS_PER_FRAME,     # 120
                lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                window_frames=72,                              # attn_window
                supports_provisional=True,                     # update_cache=1 writes is_pred
            ),
            KVStreamSpec(
                name="action",
                tokens_per_frame=_ACTION_TOKENS_PER_FRAME,     # 16
                lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                window_frames=72,
                supports_provisional=True,
            ),
        ),

        # Three phases per control step. The video loop must fully complete before the action
        # loop starts: the 26th video forward writes provisional K/V that all 51 action forwards
        # read (wan_va_server.py:504-508 then :542-548). That is a hard barrier, declared here
        # via depends_on so the scheduler plans around it instead of discovering it.
        phases=(
            PhaseSpec(
                name="kv_refresh", nfe=2,
                reads=frozenset({"video", "action"}),
                writes=frozenset({"video", "action"}),
                commit_steps=frozenset({0, 1}),     # BOTH forwards run update_cache=2
            ),
            PhaseSpec(
                name="video", nfe=26,               # num_inference_steps 25, +1 padded timestep
                reads=frozenset({"video", "action"}),
                writes=frozenset({"video"}),
                commit_steps=frozenset({25}),       # last_step -> update_cache=1
                truncatable=True,                   # video_exec_step already exists, set to -1
                min_nfe=1,
                depends_on=("kv_refresh",),
            ),
            PhaseSpec(
                name="action", nfe=51,              # action_num_inference_steps 50, +1 padded
                reads=frozenset({"video", "action"}),
                writes=frozenset({"action"}),
                commit_steps=frozenset({50}),
                truncatable=True,
                min_nfe=1,
                depends_on=("video",),
            ),
        ),

        # guidance_scale=5 on video, action_guidance_scale=1 on action. The action stream's
        # combine takes the else branch and keeps [:1] (wan_va_server.py:552-555), i.e. its
        # negative branch is computed and discarded. Declared as a FACT; CFGBranchElision is
        # what turns it into an optimization.
        guidance={
            "video": GuidanceRule(mode=GuidanceMode.CFG, scale=5.0, batchable=True),
            "action": GuidanceRule(mode=GuidanceMode.POSITIVE_ONLY, scale=1.0, batchable=True),
        },

        # Episode-constant conditioning. The instruction is encoded once in _reset
        # (wan_va_server.py:421-435) and never changes within an episode, yet cross-attention
        # gets attn_caches=None (model.py:331) so its k/v projections over the 512-token text
        # embedding are recomputed in all 30 layers on all 77 forwards.
        purity=(
            PurityKey(artifact="text_kv", fields=("prompt",), scope=KVLifetime.EPISODE),
            PurityKey(artifact="negative_text_kv", fields=("negative_prompt",),
                      scope=KVLifetime.EPISODE),
        ),

        # The predicted video is never consumed by the RoboTwin client — it asks only for
        # actions. `_infer` returns latents that the caller drops (wan_va_server.py:623-624).
        obs_decode_modules=("vae.decoder",),
        # A cycle folds in the window observed while the last action chunk executed: 8 frames per
        # camera, uint8 HWC, as a list under "obs". _ControlLoop conditions the action forward on the
        # newest of them and advances the ring with the whole window.
        observation=ObservationSpec(
            fields=tuple(ObservationField(key=f"observation.images.{c}", shape=(240, 320, 3),
                                          dtype="uint8")
                         for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")),
            history=8, batched=False, frames_key="obs", conditioning=("prompt",)),

        notes={
            "attn_mode": "torch (custom_sdpa); forced by the server and by transformer/config.json",
            "kv_pool": "9792 slots, grows 272 tokens/cycle, saturates ~cycle 36, 6.72 GiB",
            "measured_stock_cycle_ms": "8881 on idle H100 = 32 actions = 3.6 Hz",
            "known_sync": "model.py:451 mask.nonzero() per layer per forward",
            "known_gather": "model.py:452-453 key_pool[:, valid] re-gathers the whole pool",
        },
    )


#: Declared guidance mode -> the server's per-stream guidance scale attribute.
_GUIDANCE_ATTR = {"video": "guidance_scale", "action": "action_guidance_scale"}


def apply_declared_guidance(cfg, guidance) -> dict:
    """Make `execution.guidance` actually take effect. Returns what was applied, for the log.

    Before this, the declaration's guidance block was parsed, echoed by `describe()`, and then
    ignored: nothing wrote `cfg.guidance_scale`. A checkpoint declaring `video: positive_only` --
    which is what any guidance-distilled student declares -- was served with the CFG combine still
    applied, so it produced wrong actions AND paid the doubled batch it had been trained to avoid.
    Silently serving something other than what the checkpoint declares is the failure this whole
    two-namespace design exists to prevent, and it was happening in the shipped path.

    The declaration names the MODE; the scale is a model fact and stays in the server config. So
    `positive_only` pins the scale to 1.0 (the server's own `guidance_scale > 1` test is what turns
    CFG off), `cfg` leaves the model's configured scale alone, and an explicit numeric scale wins if
    a checkpoint declares one.
    """
    applied = {}
    for stream, mode in dict(guidance or {}).items():
        attr = _GUIDANCE_ATTR.get(stream)
        if attr is None or not hasattr(cfg, attr):
            continue
        scale = None
        if isinstance(mode, (int, float)) and not isinstance(mode, bool):
            scale = float(mode)
        elif isinstance(mode, dict):                      # {"mode": "cfg", "scale": 5.0}
            if mode.get("scale") is not None:
                scale = float(mode["scale"])
            elif str(mode.get("mode", "")).lower() in ("positive_only", "none"):
                scale = 1.0
        elif str(mode).lower() in ("positive_only", "none"):
            scale = 1.0
        if scale is not None:
            setattr(cfg, attr, scale)
            applied[stream] = scale
    return applied


# --- observation geometry: declared per checkpoint, never guessed --------------------------------

#: The upstream config fields that decide WHAT the model looks at: which cameras, at what
#: resolution, composited how. The names match wan_va/configs/*.py verbatim, so a training config
#: can be transcribed field-for-field into the `execution` block of instinctflash.json.
GEOMETRY_KEYS = ("obs_cam_keys", "height", "width", "env_type")


def _normalized_geometry(g: Mapping) -> dict:
    return {
        "obs_cam_keys": [str(k) for k in g["obs_cam_keys"]],
        "height": int(g["height"]),
        "width": int(g["width"]),
        "env_type": str(g["env_type"]),
    }


def _upstream_va_configs() -> Mapping:
    """The upstream VA_CONFIGS registry, without importing the torch-heavy server module.

    `wan_va/configs/` imports easydict and nothing heavier, so an IFL_CFG lookup stays possible in
    an interpreter that could never host the model.
    """
    import sys

    from instinctflash.runtime.lingbot_install import resolve_lingbot_root

    wan = os.path.join(resolve_lingbot_root(None), "wan_va")
    if wan not in sys.path:
        sys.path.insert(0, wan)
    from configs import VA_CONFIGS  # noqa: PLC0415  (importable only after the insert above)
    return VA_CONFIGS


def resolve_observation_geometry(execution, *, va_configs: Mapping | None = None,
                                 environ: Mapping | None = None) -> tuple[dict, str]:
    """Where this checkpoint's cameras and frame geometry come from. Returns (geometry, source).

    Resolution order, per key in `GEOMETRY_KEYS`:

        1. the checkpoint's own DECLARATION (`execution.obs_cam_keys` / `height` / `width` /
           `env_type` in instinctflash.json)
        2. the `IFL_CFG` environment variable, naming an upstream `VA_CONFIGS` entry
        3. FAIL LOUD.

    There is deliberately no silent default. This used to be a plain VA_CONFIGS lookup that fell
    back to "robotwin", so a custom checkpoint served without an
    env var no document mentioned ran under another robot's camera keys, resolution and
    compositing — conditioning entirely wrong, with no warning, while `--serve.smoke` printed the
    robotwin camera names as what the model "expects". Guessed geometry is a wrong action, not a
    convenience.
    """
    environ = os.environ if environ is None else environ
    extra = dict(getattr(execution, "extra", None) or {})
    # A scaffolded declaration carries the literal "FILL_ME" for every geometry fact it refused
    # to guess (descriptors/scaffold.py). That sentinel is "not declared", never a value — so an
    # unfilled scaffold gets this function's loud declare-or-IFL_CFG error instead of feeding
    # int("FILL_ME") into the server config.
    declared = {k: extra[k] for k in GEOMETRY_KEYS
                if extra.get(k) is not None and extra.get(k) != "FILL_ME"}
    missing = [k for k in GEOMETRY_KEYS if k not in declared]
    if not missing:
        src = str(getattr(execution, "source", "") or "instinctflash.json")
        return _normalized_geometry(declared), f"declaration ({src})"

    cfg_name = environ.get("IFL_CFG")
    if cfg_name:
        cfgs = _upstream_va_configs() if va_configs is None else va_configs
        if cfg_name not in cfgs:
            raise RuntimeError(
                f"IFL_CFG={cfg_name!r} names no upstream LingBot-VA config. Known names: "
                f"{', '.join(sorted(cfgs))}. The registry is wan_va/configs/__init__.py.")
        cfg = cfgs[cfg_name]
        merged = dict(declared)
        for k in missing:
            if not hasattr(cfg, k):
                raise RuntimeError(f"upstream config {cfg_name!r} carries no {k!r}; it cannot "
                                   f"supply this checkpoint's observation geometry")
            merged[k] = getattr(cfg, k)
        source = f"IFL_CFG={cfg_name}"
        if declared:
            source += f", with the declaration overriding {sorted(declared)}"
        return _normalized_geometry(merged), source

    model_id = getattr(execution, "model_id", "") or "this checkpoint"
    raise RuntimeError(
        f"{model_id}: no observation geometry. The wan_va backbone cannot serve a checkpoint "
        f"without knowing which cameras it was trained on, at what resolution, and how the views "
        f"are composited — and it refuses to guess: the old silent 'robotwin' default ran custom "
        f"checkpoints under another robot's cameras with no warning.\n\n"
        f"Missing declaration keys: {', '.join(missing)}.\n\n"
        f"Fix (preferred): declare them in the package's instinctflash.json, inside the "
        f"\"execution\" block, copied from YOUR training config (wan_va/configs/*.py). The "
        f"robotwin values, as an example of the shape, not a default:\n\n"
        f'    "obs_cam_keys": ["observation.images.cam_high",\n'
        f'                     "observation.images.cam_left_wrist",\n'
        f'                     "observation.images.cam_right_wrist"],\n'
        f'    "height": 256,\n'
        f'    "width": 320,\n'
        f'    "env_type": "robotwin_tshape"\n\n'
        f"Or, for a one-off run, set the environment override IFL_CFG=<name> to adopt an upstream "
        f"VA_CONFIGS entry wholesale (e.g. IFL_CFG=robotwin, IFL_CFG=fans). A declared key always "
        f"wins over IFL_CFG.")


class _ControlLoop:
    """One loopable control cycle over wan_va's two-call server protocol.

    The server wants `infer(obs)` to get an action, then a second `infer(compute_kv_cache=True,
    state=<executed action>, obs=<frames observed while it executed>)` to advance the KV ring. Call
    `infer` twice without the commit and the ring never moves, so the third temporal tap is missing
    and the model raises a conv-size error several frames later. That protocol is real, and it is
    also nobody's business but this adapter's -- which is why it stops here.

    THE COMMIT IS DEFERRED, and that is physics rather than convenience. The ring is advanced with
    the frames observed *while the action chunk was executing*, so those frames do not exist yet when
    the action is produced. The runtime hands us the action through `commit()`; we hold it, and fold
    it in on the next `predict()` when the caller brings the frames that resulted from it.

    So a caller loops:

        while not done:
            action = runtime.predict({"obs": frames_observed_since_last_call, "prompt": task})

    and never learns that a KV ring exists.
    """

    #: Frames the ring consumes per cycle. `FIRST` is halved because cycle 0 prepends `init_latent`,
    #: which every profiling harness in `eval/` also does. Adapter knowledge, deliberately not in the
    #: checkpoint declaration: it describes how this backbone consumes observations, not an
    #: execution fact a planner could act on.
    FRAMES_PER_CYCLE = 8
    FRAMES_FIRST_CYCLE = 4

    def __init__(self, server, cameras: tuple[str, ...]):
        self._server, self._cameras = server, cameras
        self._pending_action = None
        self._cycles = 0

    # -- what the runtime calls -------------------------------------------------------------------
    def reset(self, **conditioning):
        self._server.infer(dict(reset=True, prompt=conditioning.get("prompt"),
                                save_visualization=False))
        self._pending_action = None
        self._cycles = 0

    def predict(self, observation):
        frames = self._frames(observation)
        if self._pending_action is not None:
            self._advance_ring(frames, self._pending_action)
            self._pending_action = None
        # The two calls want DIFFERENT observations and conflating them is a real error: the ring
        # advance encodes the whole window of frames observed during execution, while the action
        # forward conditions on the CURRENT frame only. Feeding the window to both makes the VAE
        # fail with "size of tensor a (8) must match tensor b (4)", because its temporal shortcut is
        # built for a single frame here.
        payload = dict(observation)
        if frames:
            payload["obs"] = frames[-1:]
        out = self._server.infer(payload)
        self._cycles += 1
        return out

    def commit(self, observation, action):
        """Remember the executed action. It is folded in on the next `predict`; see the class note."""
        self._pending_action = action

    def close(self):
        self._server = None

    # -- private ----------------------------------------------------------------------------------
    def _frames(self, observation) -> list:
        obs = observation.get("obs") if isinstance(observation, dict) else observation
        if obs is None:
            return []
        return list(obs) if isinstance(obs, (list, tuple)) else [obs]

    def _advance_ring(self, frames: list, action) -> None:
        need = self.FRAMES_FIRST_CYCLE if self._cycles <= 1 else self.FRAMES_PER_CYCLE
        if len(frames) < need:
            raise ValueError(
                f"cycle {self._cycles + 1} of this episode needs at least {need} observed frames to "
                f"advance the model's state, but the observation carried {len(frames)}.\n\n"
                f"A {self.__class__.__name__} control cycle consumes the frames observed WHILE the "
                f"previous action chunk was executing. Pass them as a list:\n\n"
                f"    runtime.predict({{'obs': [frame_1, ..., frame_{need}], 'prompt': task}})\n\n"
                f"where each frame is a dict keyed by {list(self._cameras)}. Padding or repeating "
                f"frames here would keep the shapes legal and silently corrupt the episode, so this "
                f"raises instead.")
        self._server.infer(dict(obs=frames[-need:], compute_kv_cache=True, imagine=False,
                                save_visualization=False, state=action))


class LingBotVA:
    """`BackendAdapter` for LingBot-VA posttrained on RoboTwin 2.0.

    The upstream server is patched at runtime rather than forked. That keeps `git status` in
    the lingbot-va checkout clean, keeps every variant one flag away from stock, and keeps the
    bit-exactness gate (`eval/lingbot_va_robotwin/probe_bitexact.py`) meaningful.
    """

    model_id = "lingbot-va-posttrain-robotwin"

    def __init__(self, lingbot_root: str | None = None):
        #: None means "resolve from LINGBOT_ROOT at serve time", so constructing an adapter
        #: never touches the filesystem — `spec()` must work on a box with no checkpoint.
        self.lingbot_root = lingbot_root

    def spec(self) -> AdapterSpec:
        return lingbot_va_spec()

    def observation_contract(self, checkpoint) -> tuple[ObservationSpec, str]:
        """What `predict` expects FOR THIS CHECKPOINT, and where that answer came from.

        `spec().observation` is the adapter's static robotwin example. A checkpoint that declares
        its own geometry gets a contract built from ITS camera keys and resolution, so
        `Runtime.observation` and `--serve.smoke` stop printing another robot's camera names.
        The per-frame shape is the model's native (height, width); the server resizes incoming
        frames to it, so that is the resolution to supply, not a hard requirement.

        Raises the same loud declare-or-IFL_CFG error as serving would, which is the point: the
        smoke test must not teach the wrong cameras.
        """
        import dataclasses

        geometry, source = resolve_observation_geometry(checkpoint.execution)
        fields = tuple(
            ObservationField(key=k, shape=(geometry["height"], geometry["width"], 3),
                             dtype="uint8")
            for k in geometry["obs_cam_keys"])
        return dataclasses.replace(lingbot_va_spec().observation, fields=fields), source

    def install(self, server_module: object, plan) -> Sequence[str]:
        # Imported here, not at module scope: the runtime layer needs torch, and reading a
        # model's declarations must not.
        from instinctflash.runtime.lingbot_install import install_plan

        return install_plan(server_module, server_module.VA_Server, plan)

    def serve(
        self,
        plan,
        port: int,
        config_name: str = "robotwin",
        save_root: str | None = None,
        deterministic_seed: int | None = None,
    ):
        """Import the upstream server, install `plan`, and run it on `port`.

        Installation happens BEFORE `run()` because `run()` is what builds the model, and
        `fsdp_elision` replaces the bound `_configure_model` the build calls through.
        """
        from instinctflash.runtime.lingbot_install import (
            import_lingbot_server,
            install_deterministic_seed,
        )

        server_module = import_lingbot_server(self.lingbot_root)
        applied = list(self.install(server_module, plan))
        if deterministic_seed is not None:
            applied += install_deterministic_seed(server_module, deterministic_seed)

        print("=" * 72, flush=True)
        print(f"InstinctFlash serving {self.model_id} on :{port}", flush=True)
        print(f"  plan tier : {plan.tier().name}", flush=True)
        print(f"  applied   : {applied if applied else ['STOCK BASELINE']}", flush=True)
        print("=" * 72, flush=True)

        args = _ServerArgs(config_name=config_name, port=port, save_root=save_root)
        server_module.init_logger()
        return server_module.run(args)


    # -- placement hooks, read by instinctflash.runtime.execution ------------------------------------
    #
    # These are OPTIONAL methods that `Runtime` looks up with getattr. They are not part of the
    # `BackendAdapter` Protocol and adding them here does not widen it: an adapter without them
    # simply cannot offer that placement, and the facade says so in a message that explains the fix.

    BACKBONE = "wan_va"

    #: The subdirectories this backbone's loader expects. The package supplies the trainable one;
    #: the rest come from `execution.base_weights`.
    TRAINABLE_COMPONENT = "transformer"
    FROZEN_COMPONENTS = ("vae", "text_encoder", "tokenizer")

    @classmethod
    def materialize(cls, checkpoint) -> str:
        """Compose a loadable checkpoint tree from the package plus its `base_weights` pointer.

        This is what makes "reference the frozen stack by repo id" work in practice. The published
        package carries the trainable transformer flat at its root; the upstream loader wants
        `transformer/ vae/ text_encoder/ tokenizer/`. So compose a directory of SYMLINKS -- no bytes
        are copied, and the 14.2 GB frozen stack is shared by every checkpoint that points at it.

        Knowing that this backbone wants those four subdirectories is ADAPTER knowledge. The runtime
        only knows there is a pointer; it has no idea what a VAE is, and must not.
        """
        import os
        from pathlib import Path

        pkg = Path(checkpoint.path)
        base = (checkpoint.execution.extra or {}).get("base_weights")
        base = os.environ.get("LINGBOT_CKPT") or base
        if not base:
            raise RuntimeError(
                f"{checkpoint.model_id}: execution.base_weights is not declared and LINGBOT_CKPT is "
                f"unset, so the frozen stack cannot be resolved. This backbone needs "
                f"{', '.join(cls.FROZEN_COMPONENTS)} in addition to the packaged transformer.")
        basep = Path(base)
        if not basep.exists():
            from huggingface_hub import snapshot_download
            # ONLY the frozen components. The base repo also carries its own `transformer/`, which
            # this package REPLACES -- fetching it costs a consumer 9.5 GB they will never load.
            # Measured on a clean box: 23 GB pulled where 13 GB was needed. `allow_patterns` is
            # what keeps "reference the frozen stack by repo id" cheaper than vendoring it, which
            # is the entire argument for the pointer.
            basep = Path(snapshot_download(
                str(base), allow_patterns=[f"{c}/*" for c in cls.FROZEN_COMPONENTS]))

        # NOT inside the package. The package may be a Hugging Face snapshot directory, which is a
        # shared read-only-by-convention cache; writing a composed tree into it pollutes every other
        # consumer of that snapshot. It is also actively harmful for round-tripping: an earlier
        # version wrote `<pkg>/.instinctflash_composed/`, and a subsequent `hf upload` of that directory
        # published a SECOND 10 GB copy of the transformer into the repo, as symlink targets
        # resolved to real bytes. The composed tree is a local build artifact, so it belongs in a
        # cache keyed by what it was composed FROM.
        import hashlib
        key = hashlib.sha1(f"{pkg.resolve()}\x00{basep.resolve()}".encode()).hexdigest()[:16]
        root = Path(os.environ.get("IFL_CACHE") or
                    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "instinctflash")
        composed = root / "composed" / key

        def link_to(link: Path, target: Path, *, is_dir: bool = False) -> None:
            if link.is_symlink() or link.exists():
                if link.is_symlink() and link.resolve() == target:
                    return
                link.unlink()                       # stale, or pointing somewhere else now
            link.symlink_to(target, target_is_directory=is_dir)

        tdir = composed / cls.TRAINABLE_COMPONENT
        flat = list(pkg.glob("*.safetensors"))
        if flat:
            # published package layout: the trainable transformer sits flat at the package root
            tdir.mkdir(parents=True, exist_ok=True)
            for f in flat + list(pkg.glob("*.index.json")) + [pkg / "config.json"]:
                link_to(tdir / f.name, f.resolve())
        elif (pkg / cls.TRAINABLE_COMPONENT).exists():
            # upstream-composed layout (a declared view of the original release): the package
            # already carries transformer/ as a component; link it whole
            composed.mkdir(parents=True, exist_ok=True)
            if tdir.exists() and not tdir.is_symlink():
                import shutil
                shutil.rmtree(tdir)                 # stale flat-layout build under the same key
            link_to(tdir, (pkg / cls.TRAINABLE_COMPONENT).resolve(), is_dir=True)
        else:
            raise RuntimeError(f"{checkpoint.model_id}: no transformer weights found — neither "
                               f"flat *.safetensors at the package root nor a transformer/ "
                               f"component directory")
        for comp in cls.FROZEN_COMPONENTS:
            src = basep / comp
            if src.exists():
                link_to(composed / comp, src.resolve(), is_dir=True)
        return str(composed)

    #: What THIS model needs in the hosting interpreter. The runtime used to hardcode this list, which
    #: made it LingBot-VA's dependency set for every model in the ecosystem; it belongs here.
    HOST_REQUIRES = ("torch", "diffusers", "transformers", "safetensors")

    def can_host_in_process(self):
        """Whether this interpreter can host LingBot-VA. Falling back to a worker is correct here:
        the RoboTwin client genuinely cannot import diffusers, which is why the boundary exists."""
        from instinctflash.runtime.execution import imports_available
        return imports_available(self.HOST_REQUIRES)

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None, seed=None):
        """Build the server in THIS process and return a loopable control loop over it.

        `seed` is threaded PER REQUEST, not just per episode: `_infer` draws `torch.randn` for
        the initial video latents and action tokens with no upstream seeding, so
        `install_deterministic_seed` re-seeds each draw as seed+frame_st_id — deterministic
        across servers, while keeping the stock within-episode noise distribution.
        """
        import os

        from instinctflash.runtime.lingbot_install import import_lingbot_server

        composed = self.materialize(checkpoint)
        os.environ["LINGBOT_CKPT"] = composed
        S = import_lingbot_server(self.lingbot_root)
        # Observation geometry: declaration > IFL_CFG > fail loud, never a silent robotwin
        # default. The named config (IFL_CFG, else robotwin) stays the structural base for the
        # NON-geometry serving facts (normalization stats, action channel map, schedules), which
        # the declaration cannot express yet; the resolved geometry overrides what it looks at.
        geometry, geo_source = resolve_observation_geometry(checkpoint.execution,
                                                            va_configs=S.VA_CONFIGS)
        base_name = os.environ.get("IFL_CFG") or "robotwin"
        cfg = S.VA_CONFIGS[base_name]
        for key, value in geometry.items():
            setattr(cfg, key, value)
        print(f"InstinctFlash observation geometry: {geo_source} — cameras "
              f"{list(cfg.obs_cam_keys)}, {cfg.height}x{cfg.width}, env_type={cfg.env_type!r} "
              f"(non-geometry serving facts from upstream config {base_name!r})", flush=True)

        # Point the upstream config at the weights EXPLICITLY. Setting LINGBOT_CKPT alone worked only
        # because this machine's checkout had been hand-edited to read that variable; a clean `git
        # clone` has `wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"` hardcoded,
        # so a first user got `OSError: ... /path/to/pretrained/model/vae is not the path to a
        # directory containing a config.json` after a 20 GB download. Depending on a local edit to a
        # third-party tree is not a dependency anyone can satisfy from the documentation.
        for attr in ("wan22_pretrained_model_name_or_path", "pretrained_model_name_or_path"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, composed)
        cfg.save_root = os.environ.get("IFL_SAVE_ROOT", "/tmp/iwm_runtime")
        os.makedirs(cfg.save_root, exist_ok=True)
        # Single-GPU serving must not require torchrun: env:// rendezvous demands these four
        # variables even at world size 1, where every collective is identity. A clean shell
        # (any machine that is not a launcher-managed rank) gets the single-process defaults.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        S.init_distributed(int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0)),
                           int(os.getenv("RANK", 0)))
        cfg.rank = cfg.local_rank = 0
        cfg.world_size = 1

        n = dict(nfe or checkpoint.execution.nfe or {})
        if n:
            cfg.num_inference_steps = int(n.get("video", cfg.num_inference_steps))
            cfg.action_num_inference_steps = int(n.get("action", cfg.action_num_inference_steps))
        apply_declared_guidance(cfg, checkpoint.execution.guidance)

        # the plan must be installed BEFORE the model is built: fsdp_elision replaces the bound
        # _configure_model that the build calls through.
        applied = list(self.install(S, plan))
        if seed is not None:
            from instinctflash.runtime.lingbot_install import install_deterministic_seed
            applied += install_deterministic_seed(S, int(seed))
        server = S.VA_Server(cfg)
        print(f"InstinctFlash in-process: applied {applied or ['STOCK BASELINE']}", flush=True)
        return _ControlLoop(server, tuple(cfg.obs_cam_keys))

    def worker_command(self, checkpoint, plan, *, port, python, device=None, nfe=None,
                       seed=None):
        """How to start this model as a managed worker. Returns (argv, env-overrides).

        Reuses `serve_variant.py` -- the entry point the project already gates and measures -- rather
        than adding a second serving path that would need its own bit-exactness evidence. The flags
        come from `shipped_configuration()`, so the worker runs exactly what the registry says ships.
        """
        import os
        from pathlib import Path

        from instinctflash.verify.released import shipped_configuration

        # Geometry follows the same declaration > IFL_CFG > fail-loud rule as build_in_process.
        # The named config rides on --config-name (previously hardcoded to robotwin, so
        # IFL_CFG=fans plus worker placement silently served robotwin geometry); declared keys
        # ride on --geometry as JSON overrides, resolved by serve_variant in the worker where the
        # upstream configs actually import.
        import json as _json

        extra = dict(checkpoint.execution.extra or {})
        # the same FILL_ME-is-undeclared rule as resolve_observation_geometry
        declared = {k: extra[k] for k in GEOMETRY_KEYS
                    if extra.get(k) is not None and extra.get(k) != "FILL_ME"}
        cfg_name = os.environ.get("IFL_CFG")
        if len(declared) < len(GEOMETRY_KEYS) and not cfg_name:
            # raises the loud declare-or-IFL_CFG error before a worker is ever spawned
            resolve_observation_geometry(checkpoint.execution, va_configs={})

        iwm_root = Path(__file__).resolve().parents[2]
        serve = iwm_root / "eval" / "lingbot_va_robotwin" / "serve_variant.py"
        argv = [python, "-u", str(serve), "--config-name", cfg_name or "robotwin",
                "--port", str(port), *shipped_configuration()]
        if declared:
            argv += ["--geometry", _json.dumps(declared)]

        # The DECLARED schedule, with `nfe=` overriding it -- the same resolution order as
        # build_in_process. This used to read the override only, so a checkpoint declaring
        # nfe {video: 2, action: 4} was served by a worker at the upstream 25/50 default while the
        # identical checkpoint served in-process ran 2/4. Two placements, two behaviours, from one
        # declaration: exactly what `placement` is supposed to be invisible to.
        n = {**dict(checkpoint.execution.nfe or {}), **dict(nfe or {})}
        if n:
            argv += ["--degrade-nfe", f"{n.get('video', 2)},{n.get('action', 4)}"]

        g = dict(checkpoint.execution.guidance or {})
        if g:
            argv += ["--guidance", ",".join(f"{k}={v}" for k, v in sorted(g.items()))]

        if seed is not None:
            # the same per-request seeding as build_in_process, installed by serve_variant
            argv += ["--deterministic-seed", str(int(seed))]

        env = {"PYTHONUNBUFFERED": "1"}
        shim = os.environ.get("IFL_FA_SHIM_DIR")
        if shim:
            env["PYTHONPATH"] = shim + os.pathsep + os.environ.get("PYTHONPATH", "")
        # THE COMPOSED TREE, not the base pointer. This read `base_weights` and handed the worker a
        # Hub repo id -- so the worker loaded the BASE checkpoint's transformer and ignored the
        # published package entirely, serving different weights from the same checkpoint depending on
        # placement. `materialize()` is what composes the packaged transformer with the frozen stack,
        # and it is what the in-process path has always used.
        env["LINGBOT_CKPT"] = self.materialize(checkpoint)
        if device:
            env["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1] if ":" in device else device
        return argv, env


class _ServerArgs:
    """The three attributes upstream's `run()` reads off its argparse namespace."""

    def __init__(self, config_name: str, port: int | None, save_root: str | None):
        self.config_name = config_name
        self.port = port
        self.save_root = save_root
