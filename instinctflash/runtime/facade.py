"""`Runtime` — the whole public API.

    from instinctflash import Runtime

    runtime = Runtime.from_pretrained("robbyant/lingbot-va-posttrain-robotwin")
    action = runtime.predict(observation)

Nothing above mentions a planner, a pass, a tier, a backend or a socket, and nothing below requires
the caller to learn one. The implementation vocabulary is still there and still exported, one import
deeper, for the people who need it -- but the first call does not go through it.

WHAT `from_pretrained` DOES, in order. This sequence IS the platform pipeline, which is why the
facade does not weaken it:

    1  resolve         local path, or Hub repo id
    2  validate        refuse a directory that is not a checkpoint
    3  declare         load_declaration -> the EXECUTION block only, never provenance
    4  gate            require_servable -> refuse servable=false, without asking why
    5  resolve adapter execution.backbone -> a registered Adapter, or an error that teaches
    6  describe        adapter.spec() -> the shape of a control step
    7  plan            Optimizer().compile(spec, capabilities=checkpoint.capabilities())
    8  place           in-process, or a managed worker -- see runtime/execution.py

Steps 3-4 are why `Runtime` cannot become a way to smuggle a training method into planning: the
declaration reader does not return provenance, so there is nothing for step 7 to branch on.

NO FAST/QUALITY PRESETS. An operating point is a descriptor delta, so it is either a second published
checkpoint with its own `execution.nfe`, or an explicit `nfe=` override of a field the checkpoint
already declares. A preset table inside `Runtime` would be a per-checkpoint tuning table living in the
runtime, which is the branch this whole architecture exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from instinctflash.descriptors.package import Checkpoint, from_pretrained as _load_package
from instinctflash.runtime.execution import ExecutionBackend, choose_backend


class UnknownBackboneError(RuntimeError):
    """Raised when `execution.backbone` names no registered adapter. The message is the product."""


class Runtime:
    """A loaded, planned, ready-to-run checkpoint."""

    def __init__(self, checkpoint: Checkpoint, adapter, plan, backend: ExecutionBackend,
                 *, placement_reason: str = "", precision: str = "native",
                 tier_ceiling: str | None = None, nfe: Mapping[str, int] | None = None,
                 step_cache=None):
        self._checkpoint, self._adapter, self._plan = checkpoint, adapter, plan
        self._backend, self._placement_reason = backend, placement_reason
        self._precision = precision
        self._tier_ceiling = tier_ceiling or ("numeric" if precision == "fp8" else "bitexact")
        self._requested_nfe = dict(nfe or {})
        self._step_cache = (step_cache if step_cache is not None
                            else getattr(plan, "resolved_step_cache", None))

    # -- loading ---------------------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        revision: str | None = None,
        nfe: Mapping[str, int] | None = None,
        device: str | None = None,
        placement: str = "auto",
        precision: str = "native",
        step_cache: str | None = None,
        strict: bool = True,
        tier_ceiling: str | None = None,
        exclude_passes: tuple[str, ...] | list[str] = (),
        startup_timeout_s: float = 900.0,
        seed: int | None = None,
    ) -> "Runtime":
        """Load a checkpoint and return a runtime handle.

        `revision`  commit, branch or tag. Pin it when a number has to be reproducible.
        `nfe`       explicit override of the checkpoint's declared forwards-per-stream. Not a preset.
        `seed`      seed the RNGs the model draws noise from, per episode; adapters that can
                    thread it deeper seed per request (wan_va seeds every `_infer` draw). Two
                    runtimes with the same seed and the same inputs produce comparable outputs —
                    which is what makes value-for-value A/B possible at all, since stock serving
                    is unseeded and two stock servers already disagree. None keeps stock behaviour.
        `device`    None lets the adapter choose.
        `placement` 'auto' | 'in_process' | 'worker' | 'engine'. WHERE the model runs, not WHAT it
                    is; 'auto' is right unless you are deliberately isolating the model.
        `precision` 'native' preserves checkpoint arithmetic; 'fp8' explicitly requests FP8
                    and defaults the tier ceiling to NUMERIC. Unsupported FP8 requests refuse.
                    NUMERIC permission alone never enables FP8. Neither option changes nfe.
        `step_cache` 'dynamic' explicitly selects DreamZero approximate prediction reuse and
                    requires a behavioral ceiling. 'checkpoint' restores declaration options,
                    ignoring legacy schedule environment variables. None retains legacy options.
        `strict`    False downgrades the servable refusal to a warning, for inspection tooling.
        `tier_ceiling`     the strongest accuracy claim the plan may spend: 'bitexact', 'numeric',
                           or 'behavioral'. Raising it is a claim budget, not a speed knob.
                           None means BITEXACT for native precision. A checkpoint cannot
                           silently widen it; required numeric transforms need an explicit
                           numeric ceiling. FP8 requests default to numeric permission.
        `exclude_passes`   pass names to drop via `Plan.without` — a CALLER EXCLUSION the runtime
                           honors everywhere, including the engine placement (it cannot be
                           resurrected; see tests/test_engine_exclusion.py).
        `startup_timeout_s` worker-placement only: how long a cold load may take before the spawn
                           is declared dead. Raise it for cold 10 GB loads.
        """
        if placement not in {"auto", "in_process", "worker", "engine"}:
            raise ValueError(
                f"placement must be one of auto, in_process, worker, engine; got {placement!r}")
        from instinctflash.runtime.precision import resolve_precision
        tier_ceiling = resolve_precision(precision, tier_ceiling, placement)
        ckpt = _load_package(model_id_or_path, revision=revision, require_servable=strict)
        adapter, plan, _ = _compile_declaration(
            ckpt, nfe=nfe, device=device, tier_ceiling=tier_ceiling,
            exclude_passes=exclude_passes, step_cache=step_cache, placement=placement,
        )

        resolved = getattr(plan, "resolved_step_cache", None)
        backend, why = choose_backend(
            placement, adapter, ckpt, plan, device=device, nfe=nfe, precision=precision,
            startup_timeout_s=startup_timeout_s, seed=seed,
            **({"step_cache": resolved} if resolved is not None else {}),
        )
        return cls(ckpt, adapter, plan, backend, placement_reason=why, precision=precision,
                   tier_ceiling=tier_ceiling, nfe=nfe, step_cache=resolved)

    # -- using -----------------------------------------------------------------------------------
    def predict(self, observation: Mapping[str, Any], *, executed_action: Any = None) -> Any:
        """One control cycle: observation in, action out. Call it in a loop.

        `executed_action` reports what the robot ACTUALLY did, when a safety filter or low-level
        controller changed it. Omit it and the runtime assumes the returned action was executed.
        Models that carry no per-cycle state ignore it entirely.
        """
        return self._backend.predict(observation, executed_action=executed_action)

    def reset(self, **conditioning: Any) -> None:
        """Start a new episode on this runtime. Conditioning is whatever the checkpoint needs.

        One runtime holds one active episode. Use `episode()` for a scoped handle
        to that same state; use separate runtimes for independent observation streams.
        """
        self._backend.reset(**conditioning)

    def episode(self, **conditioning: Any) -> "Episode":
        """An explicit episode handle: `with runtime.episode(prompt=...) as ep: ep.predict(obs)`.

        This resets the runtime's one backend state. Handles do not isolate or
        snapshot episodes: opening another handle replaces the state used by an
        earlier handle. Finish each episode before opening the next, and serialize
        predict/reset/close calls on each runtime.
        """
        self._backend.reset(**conditioning)
        return Episode(self, conditioning)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- inspecting: present, documented, never required -----------------------------------------
    @property
    def precision(self) -> str:
        """Selected arithmetic policy; not an accuracy or bit-exactness certificate."""
        return self._precision

    @property
    def backend_stats(self) -> dict:
        """Snapshot observed local statistics without loading a model or contacting a worker.

        The envelope reports available, not_loaded or unsupported; available
        statistics are detached copies. Errors from a loaded statistics provider
        propagate. Selected execution policy remains in ``execution_policy``.
        """
        from instinctflash.runtime.telemetry import backend_stats_snapshot
        return backend_stats_snapshot(self._backend)

    @property
    def execution_policy(self) -> dict:
        """Selected permissions and schedule; independent of a measured equality certificate."""
        from copy import deepcopy
        checkpoint = getattr(self, "_checkpoint", None)
        declared = dict(getattr(getattr(checkpoint, "execution", None), "nfe", None) or {})
        effective = {**declared, **getattr(self, "_requested_nfe", {})}
        changed = {name: {"checkpoint": declared.get(name), "selected": value}
                   for name, value in effective.items() if declared.get(name) != value}
        plan = getattr(self, "_plan", None)
        transform_tier = plan.tier().name if plan is not None else "UNVERIFIED"
        category = ("OPERATING-POINT" if changed or transform_tier == "BEHAVIORAL"
                    else "FP8" if self.precision == "fp8"
                    else "NUMERIC" if transform_tier == "NUMERIC"
                    else "BITEXACT" if transform_tier == "BITEXACT" else "UNVERIFIED")
        return {"category": category, "precision": self.precision,
                "tier_ceiling": getattr(self, "_tier_ceiling", None),
                "transform_tier": transform_tier, "nfe": effective,
                "changed_schedule": changed,
                "step_cache": (self._step_cache.to_dict()
                               if getattr(self, "_step_cache", None) is not None else None),
                "schedule_options": {r.name: deepcopy(r.params["execution_schedule"])
                                     for r in getattr(plan, "applied", [])
                                     if "execution_schedule" in r.params},
                "claim_scope": "selected checkpoint-relative transforms; not an accuracy certificate"}

    @property
    def model_id(self) -> str:
        return self._checkpoint.model_id

    @property
    def checkpoint(self) -> Checkpoint:
        return self._checkpoint

    @property
    def plan(self):
        """The optimization plan, read-only. What was applied to these weights, and why."""
        return self._plan

    @property
    def observation(self):
        """What `predict` expects, declared by this backbone. `None` if it declares nothing.

        A user who arrives with only a Hub id needs this. `predict(observation)` takes a dict in the
        model's own format, and the formats genuinely differ -- LingBot-VA wants eight frames as a list
        under one key, pi05 wants three cameras and a state vector under flat keys -- so without a way
        to ask, the only place the contract exists is adapter source. That was the state of things:
        even our own CLI reached into `runtime._adapter.spec()` to build a smoke-test observation,
        which is the runtime admitting the public API was missing something its own tooling needed.

        `.describe()` is the human answer, `.example()` builds a correctly-shaped zero-filled dict.

        The CLI got here the hard way. It first branched on `notes["family"] == "vla"` and hardcoded
        camera names, tensor shapes and a history of 8 -- model-specific knowledge sitting in the
        product surface, and wrong in three separate ways for the next family that arrived. Declaring
        the contract fixed the branch; exposing it here fixes the private-attribute reach that the
        fix left behind.

        An adapter that resolves the contract PER CHECKPOINT (LingBot-VA's camera keys and frame
        geometry are declaration facts, not adapter constants) implements
        `observation_contract(checkpoint) -> (ObservationSpec, source)`; the static `spec()` is the
        fallback. Such an adapter may raise here -- with the same actionable error serving would
        raise -- rather than describe another robot's cameras.
        """
        return self._observation_contract()[0]

    @property
    def observation_source(self) -> str:
        """Where the observation contract came from: the checkpoint's declaration, an environment
        override, or the adapter's static declaration. `--serve.smoke` prints it so a wrong
        geometry source is visible before anyone trusts the 'expects' line."""
        return self._observation_contract()[1]

    def _observation_contract(self):
        fn = getattr(self._adapter, "observation_contract", None)
        if callable(fn):
            return fn(self._checkpoint)
        spec = getattr(self._adapter, "spec", None)
        obs = getattr(spec(), "observation", None) if callable(spec) else None
        return obs, "the adapter's static declaration"

    def explain(self) -> str:
        """Everything a bug report needs, in one string."""
        ex = self._checkpoint.execution
        out = [f"InstinctFlash runtime for {ex.model_id!r}",
               f"  package     : {self._checkpoint.path}",
               f"  backbone    : {ex.backbone}",
               f"  servable    : {ex.servable}",
               f"  precision   : {self._precision}",
               "  evidence    : no closed-loop certificate verified for this live execution; plan tiers are transformation claims",
               f"  placement   : {self._placement_reason}",
               f"  capabilities: {', '.join(sorted(self._checkpoint.capabilities()))}",
               "", self._plan.explain()]
        return "\n".join(out)

    def __repr__(self) -> str:
        return f"<Runtime {self._checkpoint.model_id!r} backbone={self._checkpoint.execution.backbone!r}>"


class Episode:
    """A scoped handle to the runtime's current rollout, without independent state.

    Created by `Runtime.episode()`. Opening another episode on the same runtime
    resets the shared backend; an older handle does not retain its previous state.

    The verbs stop here deliberately. There is no `step()` -- ambiguous between a denoising step and
    a control step, and it buys nothing over `predict`. There is no `commit()` -- committing state is
    a phase inside a control cycle, and a model that needs one says so to the runtime, not to the
    user. If a future model genuinely cannot express itself as observation-in/action-out, that is the
    conversation to have then, and it should change this class rather than leak past it.
    """

    def __init__(self, runtime: "Runtime", conditioning: Mapping[str, Any]):
        self._runtime, self._conditioning, self._closed = runtime, dict(conditioning), False
        self._steps = 0

    def predict(self, observation: Mapping[str, Any], *, executed_action: Any = None) -> Any:
        if self._closed:
            raise RuntimeError("this episode is finished; start another with runtime.episode(...)")
        self._steps += 1
        return self._runtime.predict(observation, executed_action=executed_action)

    @property
    def steps(self) -> int:
        """Control cycles issued in this episode."""
        return self._steps

    def close(self) -> None:
        """End the episode. The MODEL stays loaded -- that is `Runtime.close()`."""
        self._closed = True

    def __enter__(self) -> "Episode":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Episode of {self._runtime.model_id!r} steps={self._steps}>"


def _unknown_backbone_message(ckpt: Checkpoint, registered: list[str]) -> str:
    backbone = str(ckpt.execution.backbone)
    head = (
        f"checkpoint {ckpt.model_id!r} declares\n"
        f"  execution.backbone = {ckpt.execution.backbone!r}\n"
        f"but no adapter is registered for it in this environment.\n\n"
        f"Registered backbones: {', '.join(registered) or '(none)'}\n\n")
    from instinctflash.runtime.loader import discovery_problem
    broken = discovery_problem(backbone)
    if broken:
        # The right package IS installed; its adapter failed to import. Repeating the install
        # hint here would send the user in a circle — the import error is the actionable fact.
        return head + (
            f"A package advertising this backbone is installed, but its adapter failed to "
            f"import:\n\n    {backbone} {broken}\n\n"
            f"Fix that import (usually a missing dependency in the adapter's environment) and "
            f"rerun the same command.")
    first_party = _first_party_adapter_hint(backbone)
    if first_party:
        return head + first_party
    return head + (
        f"An adapter tells InstinctFlash the SHAPE of a control step -- streams, phases, guidance. It is\n"
        f"a small Python class, it lives in your project rather than in InstinctFlash, and you register\n"
        f"it with:\n\n"
        f"    instinctflash.register({ckpt.execution.backbone!r}, MyAdapter)\n\n"
        f"Worked example : examples/tiny_wam/adapter.py\n"
        f"Why it is required, and what would remove the requirement: CHECKPOINTS.md, 'Scope'.")


def _first_party_adapter_hint(backbone: str) -> str:
    """The exact install command when the missing adapter is one WE ship, else ''.

    Before this, a pi05 checkpoint whose scaffold had just proven every field stopped with
    "write an adapter class" — advice that is wrong by omission when a finished adapter package
    sits in the repository. The hint points at the local checkout when the CLI is running from
    one (the path is verified, never assumed), and at the repository otherwise.
    """
    from instinctflash.runtime.loader import FIRST_PARTY_ADAPTER_PACKAGES
    hit = FIRST_PARTY_ADAPTER_PACKAGES.get(backbone)
    if hit is None:
        return ""
    pip_name, subdir = hit
    local = Path(__file__).resolve().parents[2] / subdir
    where = str(local) if (local / "pyproject.toml").is_file() \
        else f"<your InstinctFlash checkout>/{subdir}"
    return (
        f"The {backbone!r} adapter already ships with InstinctFlash as the package "
        f"{pip_name!r}.\nInstall it into this environment and rerun the same command:\n\n"
        f"    pip install {where}\n\n"
        f"It registers itself through the 'instinctflash.adapters' entry point; nothing else "
        f"to configure.")


def load_declaration_ref(model_id_or_path: str | Path, *, revision: str | None = None):
    """Resolve only a checkpoint declaration, never a weight snapshot.

    Returns ``(declaration, raw_document, source)``.  Hub references use ``hf_hub_download`` for
    exactly one small metadata file.  This is the path behind `describe` and the CLI's plan
    preflight, so planning no longer pays the download/device cost of constructing a `Runtime`.
    """
    from instinctflash.descriptors.checkpoint import (
        DECLARATION_FILENAMES, SCHEMA_VERSION, load_declaration,
    )

    ref = str(model_id_or_path)
    p = Path(model_id_or_path)
    decl_path: Path | None = None
    doc: dict | None = None
    if p.exists():
        if not p.is_dir():
            raise RuntimeError(f"{p}: model path must be a checkpoint directory or Hub repo id")
        for name in DECLARATION_FILENAMES:
            if (p / name).is_file():
                decl_path = p / name
                break
        if decl_path is None and (p / "delta.json").is_file():
            decl = load_declaration(p)
            # Legacy files have no namespaces.  Expose only the declaration produced by the
            # quarantined legacy reader; provenance values are intentionally not reconstructed.
            doc = {"instinctflash_schema": SCHEMA_VERSION, "execution": {
                "model_id": decl.model_id, "backbone": decl.backbone,
                "servable": decl.servable, "guidance": dict(decl.guidance),
                "nfe": dict(decl.nfe),
            }}
            return decl, doc, str(p / "delta.json")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError(
                f"{ref!r} is not a local directory and huggingface_hub is not installed, so it "
                f"cannot be resolved as a Hub repo id.") from e
        # published artifacts predate the rename; try the new name, fall back to the old
        for name in DECLARATION_FILENAMES:
            try:
                decl_path = Path(hf_hub_download(ref, name, revision=revision))
                break
            except Exception:                                    # noqa: BLE001 - try compatibility name
                continue

    if decl_path is not None:
        doc = json.loads(decl_path.read_text())
    if doc is None:
        # A repo without a declaration may still be a release we know how to serve; an in-repo
        # declaration always wins, so this is a fallback, never an override.
        from instinctflash.descriptors.known import lookup
        doc = lookup(ref)
        if doc is None:
            raise RuntimeError(
                f"{ref}: no declaration (looked for {', '.join(DECLARATION_FILENAMES)}, and it is "
                f"not a known upstream release)")

    # Reuse the real reader, including schema and forbidden-provenance-key enforcement, instead of
    # maintaining a looser CLI parser for the same document.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Path(td, "instinctflash.json").write_text(json.dumps(doc))
        decl = load_declaration(td)
    return decl, doc, str(decl_path or f"known:{ref}")


def _compile_declaration(
    ckpt: Checkpoint,
    *,
    nfe: Mapping[str, int] | None = None,
    device: str | int | None = None,
    tier_ceiling: str | None = None,
    step_cache: str | None = None,
    placement: str = "auto",
    exclude_passes: tuple[str, ...] | list[str] = (),
    world_size: int = 1,
    want_pixels: bool = False,
    probe_device: bool = True,
):
    """Compile a plan from declaration facts.  Does not inspect or load checkpoint weights.

    ``tier_ceiling=None`` is the runtime's default policy; a string is an explicit caller demand
    (see ``Runtime.from_pretrained`` and the planner's checkpoint-required decision note).
    """
    from instinctflash.runtime.loader import available_models, load as load_adapter
    try:
        adapter = load_adapter(ckpt.execution.backbone)
    except KeyError as e:
        raise UnknownBackboneError(_unknown_backbone_message(ckpt, available_models())) from e

    # Plan against the schedule that will actually run, not the model's own default. The
    # checkpoint declares `execution.nfe`; `nfe=` overrides it per stream. Without this the
    # planner priced a 79-forward cycle while a 10-forward cycle executed.
    spec = _adapter_spec_for_checkpoint(adapter, ckpt)
    from instinctflash.runtime.step_cache_policy import resolve_step_cache, annotate_step_cache_plan
    resolved_step_cache = resolve_step_cache(
        ckpt, step_cache=step_cache, tier_ceiling=tier_ceiling, placement=placement,
        family=spec.notes.get("backbone", ckpt.execution.backbone))
    if (resolved_step_cache is not None
            and (resolved_step_cache.dynamic or resolved_step_cache.fixed_steps != 8)
            and "dreamzero_schedule" in exclude_passes):
        raise ValueError("Selected DreamZero step schedule conflicts with excluded dreamzero_schedule")
    # The plan header names the CHECKPOINT being planned, not the adapter's default example.
    # `spec.model_id` is the adapter author's sample checkpoint id, so a local directory whose
    # declaration says `general-instinct/lingbot-va-fans-8000` printed "InstinctFlash plan for
    # lingbot-va-posttrain-robotwin" — a wrong name on the one line a user quotes in a report.
    declared_id = (ckpt.execution.model_id or "").strip()
    if declared_id and declared_id != spec.model_id:
        import dataclasses
        spec = dataclasses.replace(spec, model_id=declared_id)
    schedule = {**dict(ckpt.execution.nfe or {}), **dict(nfe or {})}
    if schedule:
        spec = spec.with_nfe(schedule)
    # ...and at the guidance that will actually be served. The operating point is the tuple
    # (schedule grid, per-stream guidance scale, CFG batching); a checkpoint declaring video
    # guidance off (or w=3) runs batch-1 (or a different combine) and the plan must say so.
    spec = spec.with_guidance(ckpt.execution.guidance)

    # Probe the machine, so hardware requirements are enforced rather than decorative. Probing
    # is best-effort by design: analysing a checkpoint must keep working on a laptop with no
    # torch and no GPU, and a planner that refused to run without a device would break that.
    # An unprobed device is reported in the plan, never assumed away.
    probed = None
    if probe_device:
        try:
            from instinctflash.passes.contract import DeviceProfile
            probed = DeviceProfile.probe(device)
        except Exception:                                        # noqa: BLE001 - no torch, no CUDA
            if device is not None:
                raise

    from instinctflash.descriptors.deployment import DeploymentSpec
    from instinctflash.planners.planner import Optimizer, Tier
    tiers = {"bitexact": Tier.BITEXACT, "numeric": Tier.NUMERIC,
             "behavioral": Tier.BEHAVIORAL}
    ceiling_explicit = tier_ceiling is not None
    if ceiling_explicit and tier_ceiling not in tiers:
        raise ValueError(f"unknown tier ceiling {tier_ceiling!r}; one of {sorted(tiers)}")
    plan = Optimizer(tier_ceiling=tiers[tier_ceiling or "bitexact"],
                     tier_ceiling_explicit=ceiling_explicit).compile(
        spec,
        deployment=DeploymentSpec(world_size=world_size, want_pixels=want_pixels, device=probed),
        capabilities=ckpt.capabilities(),
    )
    annotate_step_cache_plan(plan, resolved_step_cache)
    if exclude_passes:
        # A caller exclusion, honored everywhere: Plan.without marks the entries excluded, and
        # choose_backend refuses to resurrect an excluded engine_offload.
        plan = plan.without(*exclude_passes)
    if resolved_step_cache is not None:
        plan.resolved_step_cache = resolved_step_cache
    return adapter, plan, probed


def _adapter_spec_for_checkpoint(adapter, checkpoint):
    """Bind an adapter's family spec to the concrete checkpoint, when the adapter supports it.

    Most backbones have one immutable shape and identity, so their existing zero-argument
    ``spec()`` remains the complete contract. A family adapter can optionally expose
    ``spec_for_checkpoint(checkpoint)`` when a declaration selects among separately measured
    upstream releases (pi05's pointer packages select ``base_weights``); the hook binds
    checkpoint facts the passes need — evidence keys, per-checkpoint geometry via the adapter's
    own ``observation_contract`` — without the planner learning any model's rules. The plan
    header ``model_id`` stays owned by the declared-id override below, never by this hook.
    """
    checkpoint_hook = getattr(adapter, "spec_for_checkpoint", None)
    return checkpoint_hook(checkpoint) if callable(checkpoint_hook) else adapter.spec()


def _snapshot_revision(path: Path) -> str | None:
    """Read the immutable Hub revision from a cache snapshot path, without resolving symlinks."""
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            revision = parts[index + 1]
            if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
                return revision
    return None


def _prepare_planning_metadata(adapter, checkpoint, repo_id: str, revision: str | None):
    """Fetch an adapter's declared metadata files for Hub-only preflight; never weights.

    Local checkpoint directories never reach this function. A resolved declaration's commit
    pins further metadata; when a built-in declaration was used, the first metadata response
    pins any following files. Adapter requests are restricted to relative metadata filenames.
    """
    from pathlib import PurePosixPath

    filenames = getattr(adapter, "PLANNING_FILES", ())
    if not isinstance(filenames, (tuple, list)):
        raise ValueError("adapter PLANNING_FILES must be a sequence of metadata filenames")
    for name in filenames:
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ":" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or PurePosixPath(name).suffix not in {".json", ".yaml", ".yml", ".toml"}):
            raise ValueError(f"unsafe or non-metadata adapter PLANNING_FILES entry: {name!r}")
    if not filenames:
        return checkpoint
    from huggingface_hub import hf_hub_download

    pinned = _snapshot_revision(Path(checkpoint.path)) or revision
    metadata_root = None
    for name in filenames:
        downloaded = Path(hf_hub_download(repo_id, name, revision=pinned))
        resolved = _snapshot_revision(downloaded)
        if resolved is not None:
            if pinned and len(pinned) == 40 and resolved != pinned:
                raise RuntimeError("Hub planning metadata did not resolve to the requested commit")
            pinned = resolved
        root = downloaded.parents[len(PurePosixPath(name).parts) - 1]
        if metadata_root is not None and root != metadata_root:
            raise RuntimeError("Hub planning metadata files resolved to different checkpoint directories")
        metadata_root = root
    from dataclasses import replace
    return replace(checkpoint, path=str(metadata_root))


def plan_declaration(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    strict: bool = True,
    nfe: Mapping[str, int] | None = None,
    tier_ceiling: str | None = None,
    precision: str = "native",
    step_cache: str | None = None,
    placement: str = "auto",
    device: str | int | None = None,
    exclude_passes: tuple[str, ...] | list[str] = (),
    world_size: int = 1,
    want_pixels: bool = False,
    probe_device: bool = True,
):
    """Return ``(checkpoint, adapter, plan, device_profile)`` without downloading weights."""
    from instinctflash.runtime.precision import resolve_precision, constrain_precision, require_fp8_plan
    tier_ceiling = resolve_precision(precision, tier_ceiling, placement)
    decl, _, source = load_declaration_ref(model_id_or_path, revision=revision)
    if strict:
        decl.require_servable(f"plan preflight for {model_id_or_path!r}")
    local_path = Path(model_id_or_path)
    is_local = local_path.is_dir()
    # Checkpoint.path is a directory everywhere else in the runtime. A declaration filename
    # here made checkpoint-specific geometry hooks look for instinctflash.json/config.json.
    metadata_root = local_path if is_local else (
        Path(source).parent if not source.startswith("known:") else Path(source))
    ckpt = Checkpoint(str(metadata_root), decl)
    if not is_local:
        from instinctflash.runtime.loader import available_models, load as load_adapter
        try:
            metadata_adapter = load_adapter(decl.backbone)
        except KeyError as error:
            raise UnknownBackboneError(_unknown_backbone_message(ckpt, available_models())) from error
        ckpt = _prepare_planning_metadata(metadata_adapter, ckpt, str(model_id_or_path), revision)
    adapter, plan, probed = _compile_declaration(
        ckpt, nfe=nfe, device=device, tier_ceiling=tier_ceiling,
        exclude_passes=exclude_passes, step_cache=step_cache, placement=placement,
        world_size=world_size, want_pixels=want_pixels, probe_device=probe_device,
    )
    constrain_precision(plan, precision)
    if precision == "fp8":
        require_fp8_plan(plan, ckpt.execution.backbone)
    return ckpt, adapter, plan, probed


def describe(model_id_or_path: str | Path, *, revision: str | None = None) -> dict:
    """What a checkpoint declares, WITHOUT downloading its weights.

    Fetches one file. On a 10 GB package that is the difference between a second and a coffee break,
    and it is how you find out whether something is servable before committing to it.

    Returns execution facts and capabilities only -- provenance is not read here either.
    """
    decl, doc, source = load_declaration_ref(model_id_or_path, revision=revision)
    ex = dict(doc.get("execution") or {})
    caps = Checkpoint(str(model_id_or_path), decl).capabilities()

    return {
        "model_id": decl.model_id,
        "backbone": decl.backbone,
        "servable": decl.servable,
        "guidance": dict(decl.guidance),
        "nfe": dict(decl.nfe),
        "output_projection": (None if decl.output_projection is None else {
            "kind": decl.output_projection.kind,
            "n_intervals": decl.output_projection.n_intervals,
            "block": decl.output_projection.block,
            "velocity_convention": decl.output_projection.velocity_convention,
            "foldable": decl.output_projection.foldable,
        }),
        "capabilities": sorted(caps),
        # Presence is safe to reveal; values stay out of the runtime and CLI result.
        "has_provenance": bool(doc.get("provenance")),
        "extra": dict(ex.get("extra", {})) or {k: v for k, v in decl.extra.items()},
        "declaration_source": source,
    }
