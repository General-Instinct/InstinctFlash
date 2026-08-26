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
                 *, placement_reason: str = ""):
        self._checkpoint, self._adapter, self._plan = checkpoint, adapter, plan
        self._backend, self._placement_reason = backend, placement_reason

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
        strict: bool = True,
        tier_ceiling: str = "bitexact",
        exclude_passes: tuple[str, ...] | list[str] = (),
        startup_timeout_s: float = 900.0,
    ) -> "Runtime":
        """Load a checkpoint and return a runtime handle.

        `revision`  commit, branch or tag. Pin it when a number has to be reproducible.
        `nfe`       explicit override of the checkpoint's declared forwards-per-stream. Not a preset.
        `device`    None lets the adapter choose.
        `placement` 'auto' | 'in_process' | 'worker'. WHERE the model runs, not WHAT it
                    is; 'auto' is right unless you are deliberately isolating the model.
        `strict`    False downgrades the servable refusal to a warning, for inspection tooling.
        `tier_ceiling`     the strongest accuracy claim the plan may spend: 'bitexact' (default),
                           'numeric', or 'behavioral'. Raising it is a claim budget, not a speed knob.
        `exclude_passes`   pass names to drop via `Plan.without` — a CALLER EXCLUSION the runtime
                           honors everywhere; an excluded pass cannot be resurrected by a placement.
        `startup_timeout_s` worker-placement only: how long a cold load may take before the spawn
                           is declared dead. Raise it for cold 10 GB loads.
        """
        if placement not in {"auto", "in_process", "worker"}:
            raise ValueError(
                f"placement must be one of auto, in_process, worker; got {placement!r}")
        ckpt = _load_package(model_id_or_path, revision=revision, require_servable=strict)
        adapter, plan, _ = _compile_declaration(
            ckpt, nfe=nfe, tier_ceiling=tier_ceiling, exclude_passes=exclude_passes,
        )

        backend, why = choose_backend(
            placement, adapter, ckpt, plan, device=device, nfe=nfe,
            startup_timeout_s=startup_timeout_s,
        )
        return cls(ckpt, adapter, plan, backend, placement_reason=why)

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

        The simple, single-robot form. For concurrent or clearly-scoped episodes use `episode()`.
        """
        self._backend.reset(**conditioning)

    def episode(self, **conditioning: Any) -> "Episode":
        """An explicit episode handle: `with runtime.episode(prompt=...) as ep: ep.predict(obs)`.

        WHY BOTH THIS AND `reset()`. `reset()` mutates one implicit episode, which is exactly right
        for one robot in one loop and is what LeRobot exposes. It has no way to say "these two
        rollouts are separate", so as soon as a fleet shares one loaded model -- the case this
        runtime exists for -- episode identity has to become a value the caller holds rather than
        hidden state the caller hopes it reset. vLLM reached the same conclusion and calls it a
        request id.
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
        """
        spec = getattr(self._adapter, "spec", None)
        return getattr(spec(), "observation", None) if callable(spec) else None

    def explain(self) -> str:
        """Everything a bug report needs, in one string."""
        ex = self._checkpoint.execution
        out = [f"InstinctFlash runtime for {ex.model_id!r}",
               f"  package     : {self._checkpoint.path}",
               f"  backbone    : {ex.backbone}",
               f"  servable    : {ex.servable}",
               f"  placement   : {self._placement_reason}",
               f"  capabilities: {', '.join(sorted(self._checkpoint.capabilities()))}",
               "", self._plan.explain()]
        return "\n".join(out)

    def __repr__(self) -> str:
        return f"<Runtime {self._checkpoint.model_id!r} backbone={self._checkpoint.execution.backbone!r}>"


class Episode:
    """One rollout. Created by `Runtime.episode()`; holds no weights and is cheap to make.

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
    return (
        f"checkpoint {ckpt.model_id!r} declares\n"
        f"  execution.backbone = {ckpt.execution.backbone!r}\n"
        f"but no adapter is registered for it.\n\n"
        f"Registered backbones: {', '.join(registered) or '(none)'}\n\n"
        f"An adapter tells InstinctFlash the SHAPE of a control step -- streams, phases, guidance. It is\n"
        f"a small Python class, it lives in your project rather than in InstinctFlash, and you register\n"
        f"it with:\n\n"
        f"    instinctflash.register({ckpt.execution.backbone!r}, MyAdapter)\n\n"
        f"Worked example : examples/tiny_wam/adapter.py\n"
        f"Why it is required, and what would remove the requirement: CHECKPOINTS.md, 'Scope'.")


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
    tier_ceiling: str = "bitexact",
    exclude_passes: tuple[str, ...] | list[str] = (),
    world_size: int = 1,
    want_pixels: bool = False,
    probe_device: bool = True,
):
    """Compile a plan from declaration facts.  Does not inspect or load checkpoint weights."""
    from instinctflash.runtime.loader import available_models, load as load_adapter
    try:
        adapter = load_adapter(ckpt.execution.backbone)
    except KeyError as e:
        raise UnknownBackboneError(_unknown_backbone_message(ckpt, available_models())) from e

    # Plan against the schedule that will actually run, not the model's own default. The
    # checkpoint declares `execution.nfe`; `nfe=` overrides it per stream. Without this the
    # planner priced a 79-forward cycle while a 10-forward cycle executed.
    spec = adapter.spec()
    schedule = {**dict(ckpt.execution.nfe or {}), **dict(nfe or {})}
    if schedule:
        spec = spec.with_nfe(schedule)

    # Probe the machine, so hardware requirements are enforced rather than decorative. Probing
    # is best-effort by design: analysing a checkpoint must keep working on a laptop with no
    # torch and no GPU, and a planner that refused to run without a device would break that.
    # An unprobed device is reported in the plan, never assumed away.
    probed = None
    if probe_device:
        try:
            from instinctflash.passes.contract import DeviceProfile
            probed = DeviceProfile.probe()
        except Exception:                                        # noqa: BLE001 - no torch, no CUDA
            pass

    from instinctflash.descriptors.deployment import DeploymentSpec
    from instinctflash.planners.planner import Optimizer, Tier
    tiers = {"bitexact": Tier.BITEXACT, "numeric": Tier.NUMERIC,
             "behavioral": Tier.BEHAVIORAL}
    if tier_ceiling not in tiers:
        raise ValueError(f"unknown tier ceiling {tier_ceiling!r}; one of {sorted(tiers)}")
    plan = Optimizer(tier_ceiling=tiers[tier_ceiling]).compile(
        spec,
        deployment=DeploymentSpec(world_size=world_size, want_pixels=want_pixels, device=probed),
        capabilities=ckpt.capabilities(),
    )
    if exclude_passes:
        # A caller exclusion, honored everywhere: Plan.without marks the entries excluded, and
        # no placement may resurrect an excluded pass.
        plan = plan.without(*exclude_passes)
    return adapter, plan, probed


def plan_declaration(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    strict: bool = True,
    nfe: Mapping[str, int] | None = None,
    tier_ceiling: str = "bitexact",
    exclude_passes: tuple[str, ...] | list[str] = (),
    world_size: int = 1,
    want_pixels: bool = False,
    probe_device: bool = True,
):
    """Return ``(checkpoint, adapter, plan, device_profile)`` without downloading weights."""
    decl, _, source = load_declaration_ref(model_id_or_path, revision=revision)
    if strict:
        decl.require_servable(f"plan preflight for {model_id_or_path!r}")
    ckpt = Checkpoint(source, decl)
    adapter, plan, probed = _compile_declaration(
        ckpt, nfe=nfe, tier_ceiling=tier_ceiling, exclude_passes=exclude_passes,
        world_size=world_size, want_pixels=want_pixels, probe_device=probe_device,
    )
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
