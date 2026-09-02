"""Backend Adapter — what a world-action model DECLARES about itself.

InstinctFlash is an optimization framework, not just a runtime. The layering is

    Backend Adapter  ->  Optimizer/Compiler  ->  Runtime

and the load-bearing idea is that the adapter states *facts*, never *optimizations*. A model
author writes "the action stream uses positive-only guidance" and "the text conditioning is a
pure function of the instruction". They do not write "skip the negative branch" or "cache the
cross-attention K/V" — the optimizer derives those. That is the difference between a runtime
you configure and a framework that makes your model fast because it understands it.

The declarations below were chosen by diffing the per-control-step execution graphs of several
world-action model families during design. **Only LingBot-VA and Cosmos3-Edge are supported**; the
others informed the shape of these fields and are not claimed as working. Two findings shaped them:

  * KV persistence is a LIFETIME, not a boolean. A prefix cache built once, committed, read from
    every denoise forward and dropped is structurally identical to LingBot-VA's
    episode-scoped stream, differing only in how long it lives. A boolean `is_stateful` would
    have excluded Cosmos3-Edge (chunk-scoped) and every VLA.
  * The clean seam is `commit_context`: everything above it is prefill, everything below is
    decode. Five of six models already draw that line in their own code without naming it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


class KVLifetime(enum.Enum):
    """How long a committed KV stream survives.

    The single most important declaration: it is what lets one optimizer serve a stateless VLA
    and an episode-scoped WAM without an `if is_vla` branch anywhere.
    """

    NONE = "none"        # GR00T: no KV persists past a forward
    CHUNK = "chunk"      # pi-0 prefix, Cosmos3-Edge text K/V, InternVLA-A1
    WINDOW = "window"    # DreamZero: N frames, hard reset at the boundary
    EPISODE = "episode"  # LingBot-VA: both streams, for the whole episode


class CommitMode(enum.Enum):
    SCRATCH = "scratch"          # written then rolled back within a step
    PROVISIONAL = "provisional"  # survives the step, invalidated at the next commit
    CONFIRMED = "confirmed"      # permanent for the declared lifetime


class GuidanceMode(enum.Enum):
    NONE = "none"                    # GR00T, InternVLA-A1, pi-0: no negative branch at all
    CFG = "cfg"                      # both branches computed AND combined
    POSITIVE_ONLY = "positive_only"  # a negative branch exists in the batch but is DISCARDED


@dataclass(frozen=True)
class KVStreamSpec:
    """One named, independently committed KV stream.

    Generalizes vLLM-Omni's `ARDiffusionKVCacheSpec` — which is the right dataclass — along the
    two axes it lacks. Theirs has a single `tokens_per_frame` (also the paged block size) and
    demotes everything else to `max_scratch_tokens_per_branch`, documented as "non-video KV (for
    example, action/state registers) that must coexist with an uncommitted video block". That
    models one video stream with registers bolted on. LingBot-VA commits action K/V permanently
    (`update_cache=2` on both streams) and attends it in every later cycle, so it needs two
    CO-EQUAL streams with different token densities.
    """

    name: str
    tokens_per_frame: int
    lifetime: KVLifetime
    commit_mode: CommitMode = CommitMode.CONFIRMED
    window_frames: int | None = None
    sink_frames: int = 0
    supports_provisional: bool = False


@dataclass(frozen=True)
class GuidanceRule:
    """Per-output-stream guidance. Declarative, because the three models that use guidance all
    implement it differently for reasons that fit in these fields."""

    mode: GuidanceMode
    scale: float = 1.0
    # True when both branches can share one forward at batch 2 (LingBot-VA); False when the
    # model cannot batch-duplicate and must run branches as separate forwards (Cosmos3-Edge).
    batchable: bool = True

    @property
    def requests_negative_branch(self) -> bool:
        """Is a negative branch computed AND combined for this stream as declared?

        The mode alone does not decide it: `cfg` at scale 1.0 is the positive branch by
        arithmetic and the LingBot server's own `guidance_scale > 1` test turns the batch
        off for it. This is the derivation behind the CFG-batching leg of an operating point
        (descriptors/guidance.py): a forward is batch-2 iff it touches a stream for which this
        is True.
        """
        return self.mode is GuidanceMode.CFG and self.scale > 1.0


@dataclass(frozen=True)
class PhaseSpec:
    """A run of forwards with homogeneous shape — the unit the optimizer schedules.

    `nfe` is mutable per control step by design: that is the whole point of declaring the loop
    instead of hiding it inside `forward()`.
    """

    name: str
    nfe: int
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()
    #: Indices within the phase whose forwards COMMIT KV. A set, not a single index: LingBot-VA's
    #: kv_refresh phase commits on BOTH of its forwards (one per stream, each update_cache=2),
    #: and a single-index field silently marked one of them elidable — which would corrupt the
    #: episode several chunks later. Empty = this phase never commits.
    commit_steps: frozenset[int] = frozenset()
    truncatable: bool = False
    min_nfe: int = 1
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class PurityKey:
    """A model ASSERTION that some conditioning artifact is a pure function of `fields`.

    This is the highest-value and riskiest declaration in the interface: it is what licenses the
    optimizer to hoist work from per-forward to per-episode. It is an assertion, so it needs a
    verifier rather than trust — see `instinctflash/verify/`.
    """

    artifact: str
    fields: tuple[str, ...]
    scope: KVLifetime
    #: True when the reference implementation ALREADY hoists this artifact, so the purity is real but
    #: there is no work left for a hoisting pass. Without this, a pass reasoning purely from scope has
    #: to guess, and it guessed wrong: pi05's plan reported conditioning_prefill as APPLYING with the
    #: reason "recomputed on all 11 forwards per control step", while upstream in fact prefills once
    #: (`sample_actions` runs the prefix with use_cache=True and threads the KV through all ten
    #: denoise steps). A pass can see the model's SHAPE from a spec; whether the implementation
    #: already exploits that shape is something only the adapter knows, so the adapter says it.
    already_hoisted: bool = False


@dataclass(frozen=True)
class ObservationField:
    """One tensor the model consumes. `shape` is per observation, with no batch dimension."""

    key: str                                   # e.g. "observation.images.top"
    shape: tuple[int, ...]                     # e.g. (3, 480, 640)
    dtype: str = "float32"                     # "float32" | "uint8" | ...
    #: Smoke-test fill values, when all-zeros is DEGENERATE for this field rather than merely
    #: uninformative. Example: GR00T's eef_9d state carries a rotation-6D whose zero vector has
    #: no orthonormalization (upstream's SVD decode diverges on it), so its declaration carries
    #: the identity frame instead. Flat values, reshaped to `shape`; None keeps zeros.
    example: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ObservationSpec:
    """What one call to `predict` expects. Declared, because callers cannot guess it.

    WHY THIS IS A DECLARATION AND NOT A CONVENTION. Three model families in this repository want three
    genuinely different things: pi05 takes three (3,224,224) float cameras plus a state vector under
    flat keys and a prompt; LingBot-VA takes EIGHT frames of (240,320,3) uint8 as a list under an `obs`
    key, plus a prompt, because its control cycle consumes the window observed while the previous
    action chunk executed. VQ-BeT, a third, declares a fixed five-observation window. None of that is
    derivable from the weights.

    Before this existed the CLI guessed: it branched on `notes["family"] == "vla"`, then hardcoded
    camera names, tensor shapes and a history of 8. That is a model-specific branch sitting in the
    product surface, which is the thing the architecture is supposed to make unnecessary -- and it
    would have been wrong for VQ-BeT in three separate ways.

    `history` is how many observations one control cycle consumes, and `frames_key` says whether they
    arrive as a list under one key or as flat keys. Both are execution facts about the model's
    interface, not model internals: they tell a caller what to collect between calls.
    """

    fields: tuple[ObservationField, ...] = ()
    #: observations consumed per control cycle. 1 for a single-observation policy, 8 for a model whose
    #: cycle folds in the window observed while the last action executed.
    history: int = 1
    #: does the model want a leading batch dimension on each tensor
    batched: bool = True
    #: when set, the `history` observations arrive as a LIST under this key rather than as flat keys
    frames_key: str | None = None
    #: non-tensor inputs the caller must supply, e.g. ("prompt",)
    conditioning: tuple[str, ...] = ()

    #: PINNED STAGING WAS TRIED HERE AND IS SLOWER. Allocating pinned host buffers from these declared
    #: shapes and copying host->pinned->device measured 0.264 ms against 0.236 ms for a plain
    #: `.to(device)` on a (1,3,480,640) float32 image. Pinned memory pays when the copy OVERLAPS
    #: compute on a separate stream; a control cycle that uploads and immediately synchronises has no
    #: overlap to exploit, so the extra host->host copy is pure cost. Do not re-add it without a
    #: measurement showing overlap. See examples/pi05_vla/measure_overhead_vs_reference.py.
    def example(self) -> dict:
        """A zero-filled observation of the declared shape. For smoke tests, never for evaluation."""
        import numpy as np

        def one(f: ObservationField):
            shape = (1, *f.shape) if self.batched and not self.frames_key else f.shape
            if f.example is not None:
                return np.asarray(f.example, dtype=np.dtype(f.dtype)).reshape(shape)
            return np.zeros(shape, dtype=np.dtype(f.dtype))

        if self.frames_key:
            frames = [{f.key: one(f) for f in self.fields} for _ in range(max(1, self.history))]
            out: dict = {self.frames_key: frames}
        else:
            out = {f.key: one(f) for f in self.fields}
        for c in self.conditioning:
            out[c] = ""
        return out

    def describe(self) -> str:
        where = f"a list of {self.history} under {self.frames_key!r}" if self.frames_key \
            else ("flat keys" + (f", {self.history} observations deep" if self.history > 1 else ""))
        parts = [f"{f.key} {tuple(f.shape)} {f.dtype}" for f in self.fields]
        tail = f"; plus {', '.join(self.conditioning)}" if self.conditioning else ""
        return f"{where}: " + "; ".join(parts) + tail


@dataclass(frozen=True)
class AdapterSpec:
    """Everything the optimizer reads. Facts only — no optimizations."""

    model_id: str
    param_bytes: int
    streams: tuple[KVStreamSpec, ...]
    phases: tuple[PhaseSpec, ...]
    guidance: Mapping[str, GuidanceRule]
    purity: tuple[PurityKey, ...] = ()
    #: modules whose output is only needed when the caller asks for predicted pixels. All four
    #: WAMs surveyed agree the observation-decode tail is optional at serving time — Cosmos3-Edge
    #: denoises 550 of 567 tokens as future video and discards them.
    obs_decode_modules: tuple[str, ...] = ()
    #: What one `predict` call expects. Empty means undeclared, which callers are told rather than
    #: left to guess -- see ObservationSpec for why this cannot be a convention.
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    notes: Mapping[str, str] = field(default_factory=dict)
    #: Per stream, where the served guidance scale came from once a checkpoint declaration has
    #: been applied (`with_guidance`): "declared", "inherited from the family default", ...
    #: Empty means the spec is the family's own statement, untouched by any declaration.
    guidance_resolution: Mapping[str, str] = field(default_factory=dict)

    def phase(self, name: str) -> PhaseSpec:
        for p in self.phases:
            if p.name == name:
                return p
        raise KeyError(name)

    def shapes_static_across_cycles(self) -> tuple[bool, str]:
        """Do tensor shapes repeat from one control cycle to the next? Derived, never declared.

        This is the property that decides whether whole-cycle graph capture can pay, and it is a fact
        about the model's declared state, not about the hardware or the pass. A stream whose lifetime
        outlives a control cycle accumulates, so the extent read on cycle N differs from cycle N-1 and
        any captured graph is invalidated; a model whose state is rebuilt every cycle presents the same
        shapes forever.

        It generalises a difference that three model families made visible. LingBot-VA declares
        EPISODE-lifetime streams and grows 152 slots per cycle, and capture measured 1.43x SLOWER
        there because of recapture. A pi05 VLA declares a CHUNK-lifetime prefix and ACT declares no
        streams at all, so both are shape-stable, which is why hand-tuned engines capture a whole
        VLA forward and win. That is a property of the model, not a smarter runtime -- and until it
        was derivable here, the only way to find out was to build the pass and measure the regression.
        """
        growing = [s.name for s in self.streams
                   if s.lifetime in (KVLifetime.WINDOW, KVLifetime.EPISODE)]
        if growing:
            return False, (f"streams {sorted(growing)} outlive a control cycle, so their extent grows "
                           f"and per-cycle shapes change")
        if not self.streams:
            return True, "no stream persists, so every cycle presents identical shapes"
        return True, (f"all streams ({', '.join(sorted(s.name for s in self.streams))}) are rebuilt "
                      f"within a control cycle, so shapes repeat")

    def with_nfe(self, nfe: Mapping[str, int]) -> "AdapterSpec":
        """This spec at a different declared step schedule. Phases are matched by name.

        WHY THE PLANNER NEEDS THIS. An adapter states the model's own schedule, which for LingBot-VA
        is 26 video + 51 action forwards. A checkpoint then declares `execution.nfe`, and the server
        is configured from it — so without this, passes reasoned about a 79-forward cycle while a
        10-forward cycle actually ran, and every profitability argument was computed against a
        configuration that never executes.

        COMMIT STEPS ARE REMAPPED, and that is the whole reason this is a method rather than a dict
        update. `commit_steps` holds indices WITHIN a phase: video declares `{25}`, meaning "the last
        of the 26 forwards is the one that writes K/V". Rewriting nfe to 2 without touching that
        leaves an index two forwards past the end, and `cfg_elision` reads `commit_steps` to decide
        which forwards may NOT have their guidance branch dropped. The committing forward would then
        be elided, its K/V never written, and the episode would go wrong several chunks later. A
        terminal commit stays terminal; anything else is clamped into range.

        Phases absent from `nfe` are untouched, so a declaration that names only the streams it cares
        about leaves fixed phases such as `kv_refresh` alone. Values below `min_nfe` are raised to it.
        """
        import dataclasses

        out = []
        for p in self.phases:
            if p.name not in nfe:
                out.append(p)
                continue
            new = max(int(nfe[p.name]), p.min_nfe)
            if new == p.nfe:
                out.append(p)
                continue
            commits = frozenset(
                (new - 1) if c == p.nfe - 1 else min(c, new - 1) for c in p.commit_steps)
            out.append(dataclasses.replace(p, nfe=new, commit_steps=commits))
        return dataclasses.replace(self, phases=tuple(out))

    def with_guidance(self, declared: Mapping[str, object] | None) -> "AdapterSpec":
        """This spec at a checkpoint's declared per-stream guidance. Streams matched by name.

        The second leg of the operating-point tuple (schedule grid, per-stream guidance scale,
        CFG batching). An adapter states the model's OWN guidance -- LingBot-VA: video cfg@5,
        action positive-only -- and a checkpoint may declare a different served point: a
        re-tuned scale (`{"video": 3}`), the negative branch off (`{"video": "positive_only"}`
        or `{"video": {"mode": "cfg", "scale": 1.0}}`), or, in the string form, the family's
        scale explicitly INHERITED. Without this the planner priced batch-2 forwards for a
        checkpoint the server ran at batch-1, and `cfg_branch_elision` reasoned about a
        negative branch that was never computed.

        Resolution rules live in `descriptors/guidance.py` (one parser for the runtime, the
        planner, the scaffold, the sweep and the control gate). Streams the declaration names
        but this spec does not model are ignored here, exactly as the serving path ignores them.
        """
        import dataclasses

        from instinctflash.descriptors.guidance import resolve

        family = {name: (rule.mode.value, float(rule.scale)) for name, rule in self.guidance.items()}
        resolved = resolve(declared, family)
        rules = dict(self.guidance)
        sources: dict[str, str] = {}
        for name, r in resolved.items():
            if name not in rules:
                continue
            rules[name] = dataclasses.replace(
                rules[name], mode=GuidanceMode(r.mode),
                scale=float(rules[name].scale if r.scale is None else r.scale))
            sources[name] = r.scale_source
        return dataclasses.replace(self, guidance=rules, guidance_resolution=sources)

    def cfg_batching(self) -> dict:
        """The CFG-batching leg of the operating point, DERIVED from guidance + phase structure.

        A stream requests its negative branch iff `GuidanceRule.requests_negative_branch`; a
        phase's forwards are batch-2 iff the phase reads or writes such a stream (LingBot-VA:
        every phase reads both streams, so video cfg@5 makes all ten 2V/4A forwards batch-2 and
        the action stream's branch is computed then discarded -- the fact `cfg_branch_elision`
        exploits). With no stream requesting one, every forward is batch-1: the campaign's
        guidance-off points. Returns per-phase branch counts and the batch-1 / batch-2 forward
        totals a latency table reports separately (bandwidth-bound devices see the difference;
        Amdahl on an H100 does not).
        """
        requesting = sorted(n for n, g in self.guidance.items() if g.requests_negative_branch)
        batched = all(self.guidance[n].batchable for n in requesting)
        per_phase: dict[str, int] = {}
        batch1 = batch2 = separate = 0
        for p in self.phases:
            touches = bool((set(p.reads) | set(p.writes)) & set(requesting))
            per_phase[p.name] = 2 if touches else 1
            if not touches:
                batch1 += p.nfe
            elif batched:
                batch2 += p.nfe
            else:
                separate += p.nfe  # branches run as separate forwards (non-batchable families)
        return {
            "negative_branch_requested_by": requesting,
            "per_phase_branches": per_phase,
            "batch1_forwards": batch1,
            "batch2_forwards": batch2,
            "separate_branch_forwards": separate,
        }

    def operating_point(self) -> str:
        """The operating point as the tuple every report must print: (schedule, guidance, batching)."""
        guidance = []
        for name, rule in sorted(self.guidance.items()):
            src = self.guidance_resolution.get(name)
            tail = f" [{src}]" if src and src != "declared" else ""
            guidance.append(f"{name}={rule.mode.value}@{rule.scale:g}{tail}")
        b = self.cfg_batching()
        total = self.total_forwards()
        if b["negative_branch_requested_by"]:
            kind = "batch-2" if b["batch2_forwards"] else "separate-branch"
            n = b["batch2_forwards"] or b["separate_branch_forwards"]
            batching = (f"{kind} on {n} of {total} declared forwards (negative branch requested by "
                        f"{', '.join(b['negative_branch_requested_by'])}); batch-1 on {b['batch1_forwards']}")
        else:
            batching = f"batch-1 on all {total} declared forwards (no stream requests a negative branch)"
        return (f"schedule {{{self.forwards_breakdown()}}} | guidance {{{', '.join(guidance) or 'none'}}} "
                f"| cfg batching {{{batching}}}")

    def total_forwards(self) -> int:
        """Every transformer forward in one control step, across all phases.

        Note this is NOT the same number as the "77 forwards" quoted throughout the LingBot-VA
        write-ups: that figure is the *denoise* loop alone (26 video + 51 action) and excludes
        the 2 kv_refresh forwards. Both counts are correct for different questions, so anything
        user-facing should say which one it means — `forwards_breakdown()` exists so a pass can
        show its work instead of quoting a bare total that disagrees with the docs.
        """
        return sum(p.nfe for p in self.phases)

    def forwards_breakdown(self) -> str:
        """`total_forwards()` with its per-phase terms, e.g. `kv_refresh=2 + video=26 + action=51`."""
        return " + ".join(f"{p.name}={p.nfe}" for p in self.phases)


class BackendAdapter(Protocol):
    """The contract a world-action model implements to be optimized by InstinctFlash."""

    def spec(self) -> AdapterSpec:
        """Immutable declarations, read once at load."""
        ...

    def install(self, server_module: object, plan: "object") -> Sequence[str]:
        """Apply an optimization plan to a concrete serving object.

        Today this patches the upstream server at runtime rather than replacing it. That is
        deliberate and temporary: it keeps every pass verifiable against the existing
        bit-exactness gate before anything is rewritten, and it keeps the vendored upstream
        tree clean so `git diff` stays reviewable.

        Returns what it actually applied, and raises on any applied pass it cannot install.
        Reporting a pass as installed when it was skipped would invalidate every number
        measured against the resulting server.
        """
        ...

    def serve(self, plan: "object", port: int, **kwargs) -> object:
        """Import this model's server, install `plan`, and start serving on `port`."""
        ...
