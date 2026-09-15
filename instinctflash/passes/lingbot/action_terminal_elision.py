"""ActionTerminalForwardElision — skip the action pred-commit forward, keep its ring bookkeeping.

The forward, and why it is dead
-------------------------------
Both denoise loops in `_infer` pad a terminal timestep t=0 (`wan_va_server.py:473, :478`) and run one
more transformer forward at it with `update_cache=1`. On that last iteration the OUTPUT is discarded:
`if not last_step:` guards every use (`:508` video, `:548` action). The forward exists to write
provisional ("pred") K/V for the current chunk into the shared ring pool.

The two loops are NOT symmetric in what happens to that write:

  * the VIDEO pred-commit (`:502-506`) is read by every action forward of the same cycle (`:542-546`),
    so the video terminal forward is live compute;
  * the ACTION pred-commit (`:542-546`, `action_mode=True and update_cache==1`) is read by nothing.
    The next transformer call in the closed-loop message order is `_compute_kv_cache`, whose FIRST
    statement is `clear_pred_cache` (`:574`), which drops every `is_pred` slot -- the 240 video slots
    and the 32 action slots -- before the real commits (`:594-602`) overwrite them.

So in closed-loop serving the action terminal forward is 1 of 10 forwards per cycle at the shipped
2V/4A operating point (1 of 8 at 2V/2A) whose value is never read. Skipping it outright is bit-exact
PRE-saturation; `eval/lingbot_va_robotwin/probe_terminal_forward.py` measured max|delta action| = 0.

Why a naive skip was REFUTED (2026-08-09), and what this pass does instead
-------------------------------------------------------------------------
Once the pool is full (9792 slots = 36 cycles x 272 committed), every write EVICTS the oldest slots
first. The action pred write's eviction is a side effect that `clear_pred_cache` does not undo:

    stock  allocate_slots (model.py:361-378)  frees the 32 oldest-id slots, then reuses them
    ring   _commit        (ring_kv.py:273-275) `count > total` advances `start` by 32

`clear_pred_cache` rolls back COUNT (drops the pred slots) but not the eviction. With the forward
skipped, the oldest action block of cycle c-36 survives into the video commit forward of cycle c,
which then attends 32 stale keys the ON arm never saw. Measured: 0 through cycle ~37, then 0.0297 /
0.0234 / 0.266 / 0.406 / 0.266 / 0.102 (probe_action_terminal.py docstring).

This pass therefore elides the FORWARD -- the 30-layer compute, ~1,860 launches -- and REPLAYS ITS
BOOKKEEPING: the slot allocation with its eviction, and the mask / id / is_pred writes, without the
K/V write. The K/V those slots would have held is never read: `clear_pred_cache` drops the slots
before the next forward, and the action commit of the same cycle overwrites them before any later
read window includes them. See `reserve_provisional_slots` for the two allocator variants (stock
mask allocator; `RingKVAddressing` interval ring). Traced to be bit-exact past the wrap; the GPU A/B
that decides the tier is `eval/lingbot_va_robotwin/probe_action_terminal_elision.py`, and the
allocator-level proof is `tests/test_ring_allocator.py::test_elision_bookkeeping`.

Two fail-closed gates, both required
------------------------------------
1. MESSAGE PATTERN. The elision is valid only if `clear_pred_cache` runs before the next transformer
   forward. The closed-loop protocol guarantees that; the vendor's offline `generate()` (`:648-655`)
   does not -- it calls `_infer` back to back with no `_compute_kv_cache` -- and our own
   `_ControlLoop` skips the ring advance when `commit()` was never called
   (`adapters/lingbot_va.py`, `predict`). So the pass never ASSUMES the pattern: the elided forward's
   arguments are stashed, its bookkeeping is applied at `clear_pred_cache` (observationally identical
   to applying it eagerly -- nothing can run in between on the valid pattern), and if any forward
   arrives first the stashed forward is MATERIALIZED (run for real, on the exact pool state it would
   have seen) and the pass disables itself for that server. The served output is then the stock
   output; only the speedup is lost.
2. CONFIG PRECONDITION. `video_exec_step == -1` (`configs/va_robotwin_cfg.py:30`). At any other value
   the video loop is truncated and its terminal step's output is consumed (`:508`); the two-loop
   structure the asymmetry argument was traced on no longer holds. The pass declines rather than
   extrapolates. Checked per `_infer` on the live `job_config`, never on an import-time copy.

Tier
----
Claimed BITEXACT: no arithmetic changes, no reduction order changes; the only difference between arms
is 32 slots of never-read K/V garbage per layer per cycle. A BITEXACT pass needs no closed-loop
certificate -- max|delta action| = 0.000e+00 through the wrap IS the certificate -- and any nonzero
delta at any cycle demotes it to NUMERIC and keeps the flag off by default.

MEASURED 2026-09-02 (H100, `--deterministic-seed`, 48 seeded cycles per episode, ABBA ON/EL/EL/ON, wrap
crossed at cycle 36 with 12 post-wrap cycles): max|delta action| = 0.000e+00 on every cycle and every
arm pair, same-arm repeats 0.000e+00, on BOTH allocators (stock mask; --ring-kv) at BOTH operating points
(2V/4A@w5 batch-2; 2V/2A@w1 batch-1); no materialization, no self-disable. Artifacts:
/home/ubuntu/iwm_results/ate_2026-09-02/ and iwm_distill/fewstep/action_terminal_elision_report.md.
Ledger: verify/released.py P010.
"""

from __future__ import annotations

import inspect

from instinctflash.adapters.base import AdapterSpec, KVLifetime
from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery, HardwareReq, Tier,
    VerifyResult,
)
from instinctflash.planners.planner import PassResult, Tier as PlanTier


#: The vendor config field the asymmetry argument depends on, and the one value it was traced at.
VIDEO_EXEC_STEP_FIELD = "video_exec_step"
VIDEO_EXEC_STEP_REQUIRED = -1


# --------------------------------------------------------------------------------------------------
# Bookkeeping without compute. Pure functions over one attention layer's cache; unit-tested against the
# real stock allocator across >40 cycles in tests/test_ring_allocator.py.
# --------------------------------------------------------------------------------------------------

def reserve_provisional_slots(attn, cache_name: str, n_tokens: int, update_cache: int = 1) -> str:
    """Replay the BOOKKEEPING of an `update_cache=1` write of `n_tokens` tokens, with no K/V write.

    Returns which allocator variant ran: "ring" (RingKVAddressing installed on this cache), "stock"
    (the vendor mask allocator), or "none" (this module has no self-attention cache to reserve in --
    cross-attention modules have `attn_caches is None`).

    Stock variant is `WanAttention.update_cache` (model.py:396-410) minus its two K/V lines: the
    allocation -- including the eviction of the oldest ids when the pool is full -- and the
    mask / id / is_pred writes. Ring variant is the metadata half of `RingKVAddressing.forward`
    (ring_kv.py:206-209) followed by `_commit` (ring_kv.py:238-294): count += n, pred += n, the
    `count > total` start advance, and the plan-buffer refresh in the graph regime. Both are the
    exact code paths the real forward takes for its side effects; neither is a re-derivation.
    """
    caches = getattr(attn, "attn_caches", None)
    if caches is None:
        return "none"
    cache = caches.get(cache_name)
    if cache is None or cache.get("k") is None:
        return "none"

    ring = cache.get("_ring")
    if ring is not None:
        total = ring["total"]
        head = (ring["start"] + ring["count"]) % total
        if head + n_tokens > total:
            raise RuntimeError(
                f"action_terminal_forward_elision: reservation of {n_tokens} slots at head {head} "
                f"would wrap a {total}-slot ring; the ring-interval model is violated and the "
                f"pass refuses to guess (see ring_kv.py VALIDITY CONDITION).")
        sl = slice(head, head + n_tokens)
        if not attn._iwm_defer_commit:
            # ring_kv.py:206-209, verbatim: in non-deferred mode forward() writes these inline.
            cache["mask"][sl] = True
            cache["id"][sl] = ring["next_id"]
            cache["is_pred"][sl] = (update_cache == 1)
            ring["next_id"] += 1
        # ring_kv.py:238-294: count/pred/start advance (+ metadata in deferred mode, + plan buffers).
        attn._iwm_commit(cache_name, n_tokens, update_cache)
        return "ring"

    # model.py:396-410 minus `cache['k'][:, slots] = key` / `cache['v'][:, slots] = value`.
    slots = attn.allocate_slots(cache_name, n_tokens)
    new_id = attn._next_cache_id(cache_name)
    cache["mask"][slots] = True
    cache["id"][slots] = new_id
    cache["is_pred"][slots] = (update_cache == 1)
    return "stock"


def action_tokens(input_dict) -> int:
    """Slots the action forward would write per layer: `'b c f h w -> b (f h w) c'` (model.py:829)."""
    x = input_dict["noisy_latents"]
    n = 1
    for d in x.shape[2:]:
        n *= int(d)
    return n


# --------------------------------------------------------------------------------------------------
# The decision logic, separated from the monkeypatch so it can be unit-tested with fakes.
# --------------------------------------------------------------------------------------------------

class ElisionController:
    """Per-transformer state machine for gate (1) and the operator/config switches.

    Events (all driven by the installed wrappers):
        on_forward(update_cache, action_mode, run, stash) -> result of the forward, or None if elided
        on_clear_pred_cache(reserve)                       -> applies the deferred bookkeeping
        on_cache_dropped()                                 -> reset/clear_cache: nothing to keep

    `armed` is set by the `_infer` wrapper for the duration of one `_infer` call, and only when the
    config precondition holds; nothing is ever elided outside an armed `_infer`. `requested` is the
    operator switch (the CLI flag / the A/B harness). `disabled_reason` is the sticky fail-closed
    state: once set, this controller never elides again.
    """

    def __init__(self, log=print):
        self.requested = True
        self.armed = False
        self.disabled_reason = ""
        self._pending = None            # (run, args, kwargs) of the elided forward, until clear_pred_cache
        self.n_elided = 0
        self.n_reserved = 0
        self.n_materialized = 0
        self.variant = ""               # "ring" | "stock", learned at the first reservation
        self._log = log
        self._announced = False

    # -- switches ----------------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.requested and self.armed and not self.disabled_reason

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def disable(self, reason: str) -> None:
        if not self.disabled_reason:
            self.disabled_reason = reason
            self._log(f"InstinctFlash action_terminal_forward_elision: DISABLED for this server -- "
                      f"{reason}", flush=True)

    # -- events ------------------------------------------------------------------------------------
    def on_forward(self, update_cache, action_mode, run, stash):
        """`run()` executes the forward now; `stash()` returns the (args, kwargs) to replay later."""
        if self._pending is not None:
            # Gate (1) fired: a forward arrived before clear_pred_cache. The elided forward's K/V
            # would have been in this forward's read window, so run it FIRST, on the pool state it
            # would have seen (nothing has touched the pool since), then never elide again here.
            p_run, p_args, p_kwargs = self._pending
            self._pending = None
            self.n_materialized += 1
            self.disable("message pattern violated: a transformer forward arrived before "
                         "clear_pred_cache (vendor generate()? a predict() without commit()?). The "
                         "elided forward was materialized on its original pool state, so the served "
                         "output is stock; only the speedup is gone.")
            p_run(*p_args, **p_kwargs)
        if self.active and update_cache == 1 and action_mode:
            self._pending = (run, *stash())
            self.n_elided += 1
            return None                 # its consumer is `if not last_step:` -- None raises if read
        return run()

    def on_clear_pred_cache(self, reserve) -> None:
        """`reserve(args, kwargs)` applies the bookkeeping of the stashed forward to every layer."""
        if self._pending is None:
            return
        _run, args, kwargs = self._pending
        self._pending = None
        self.variant = reserve(args, kwargs) or self.variant
        self.n_reserved += 1
        if not self._announced:
            self._announced = True
            self._log(f"InstinctFlash action_terminal_forward_elision: eliding the action pred-commit "
                      f"forward; bookkeeping variant = {self.variant}", flush=True)

    def on_cache_dropped(self) -> None:
        # reset: the pool the elided forward would have written is gone; nothing to reserve or replay.
        self._pending = None

    def stats(self) -> dict:
        return {
            "elided": self.n_elided, "reserved": self.n_reserved,
            "materialized": self.n_materialized, "variant": self.variant,
            "disabled_reason": self.disabled_reason, "requested": self.requested,
        }


def _unwrap(model):
    """The module that owns `blocks`: FSDP and similar wrappers proxy attributes but own none."""
    for attr in ("_fsdp_wrapped_module", "module"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "blocks"):
            return inner
    return model


def controller_of(model_or_server, create: bool = True) -> ElisionController | None:
    """The controller of a transformer (or of a server's transformer).

    Created on first use -- the pass patches CLASSES before any model exists, so the per-instance state
    has to appear lazily, and an operator toggling `requested` before the first `_infer` must get the
    same object the wrappers will use. `create=False` answers "has the pass touched this model yet".
    """
    model = getattr(model_or_server, "transformer", model_or_server)
    m = _unwrap(model)
    c = getattr(m, "_iwm_ate", None)
    if c is None and create:
        c = m._iwm_ate = ElisionController()
    return c


# --------------------------------------------------------------------------------------------------
# The pass.
# --------------------------------------------------------------------------------------------------

#: Structural preconditions on the vendor tree, asserted at install. If any is false the trace this
#: pass rests on does not describe the served code, and it refuses to load rather than guess.
_VENDOR_INVARIANTS = (
    ("update_cache=1 if last_step else 0,", 2,
     "both denoise loops commit provisional K/V on their padded terminal step"),
    ("if not last_step:", 1, "the action loop discards its terminal output"),
    ("self.transformer.clear_pred_cache(self.cache_name)", 1,
     "_compute_kv_cache drops the pred slots before committing"),
)


def check_vendor_invariants(server_module) -> None:
    """Raise unless the vendor server source still carries the structure the trace rests on.

    Reads the MODULE source, not `VA_Server._infer`'s: other installers (deterministic seed, degrade-nfe)
    wrap `_infer`, and `inspect.getsource` of a wrapper would describe the wrapper.
    """
    src = inspect.getsource(server_module)
    for needle, at_least, meaning in _VENDOR_INVARIANTS:
        if src.count(needle) < at_least:
            raise RuntimeError(
                f"action_terminal_forward_elision: vendor server no longer matches the trace "
                f"this pass rests on ({meaning}: expected {at_least}x {needle!r}). Refusing to "
                f"install rather than elide a forward whose liveness is unknown.")


class ActionTerminalForwardElision:
    name = "action_terminal_forward_elision"
    hardware = HardwareReq()            # a control-flow change; no arch requirement
    requires_capabilities = frozenset({"backbone:wan_va"})

    #: True since the wrap-crossing A/B (probe_action_terminal_elision.py, 2026-09-02) was recorded in
    #: verify/released.py as P010. While False the planner never auto-applies the pass and the CLI flag
    #: is the only way in; set it back to False if the vendor server or the ring allocator changes and
    #: the A/B has not been re-run.
    PROVEN_BITEXACT = True

    def evaluate(self, spec: AdapterSpec, deployment: DeploymentSpec) -> PassResult:
        app = self.applicability(spec, deployment.device)
        if app.applies and not self.PROVEN_BITEXACT:
            return PassResult(
                self.name, False, PlanTier.BITEXACT,
                "opt-in until the wrap-crossing A/B is recorded: enable with "
                "serve_variant --action-terminal-elision", params=app.params)
        expected = "one forward per cycle (the action pred-commit): 1 of 10 at 2V/4A, 1 of 8 at 2V/2A"
        if app.applies and deployment.device is not None:
            expected += f" (cost model: {self.expected_delta_ms(spec, deployment.device):.1f} ms)"
        return PassResult(self.name, app.applies, PlanTier[app.claimed_tier.name], app.reason,
                          params=app.params, expected_win=expected)

    def applicability(self, spec: AdapterSpec, device: DeviceProfile) -> Applicability:
        # Declared facts this rests on: an action phase that (a) writes only a provisional-capable
        # episode-scoped stream, (b) commits on its LAST step only, and (c) runs after the video
        # phase, so nothing in the same cycle can read its pred write.
        phases = {p.name: p for p in spec.phases}
        action = phases.get("action")
        if action is None:
            return Applicability(False, "no phase named 'action'; the elision is specific to the "
                                        "action pred-commit of a two-loop VA cycle")
        if action.commit_steps != frozenset({action.nfe - 1}):
            return Applicability(False, f"action phase commits on steps {sorted(action.commit_steps)}, "
                                        f"not only on its terminal step {action.nfe - 1}")
        streams = {s.name: s for s in spec.streams}
        written = [streams[n] for n in action.writes if n in streams]
        if not written or not all(s.supports_provisional and s.lifetime is KVLifetime.EPISODE
                                  for s in written):
            return Applicability(False, "the action phase does not write a provisional-capable "
                                        "episode-scoped stream, so there is no pred write to elide")
        later = [p.name for p in spec.phases if "action" in p.depends_on]
        if later:
            return Applicability(False, f"phases {later} run after the action phase within a cycle "
                                        f"and could read its pred write")
        return Applicability(
            True,
            "the action phase's terminal forward writes provisional K/V that clear_pred_cache drops "
            "before any forward reads it; its output is discarded by construction",
            discovery=Discovery.DECLARED,
            cost_term=CostTerm.PER_STEP,
            claimed_tier=Tier.BITEXACT,
            params={"phase": "action", "streams": sorted(s.name for s in written),
                    "gates": ["message_pattern", f"{VIDEO_EXEC_STEP_FIELD}=={VIDEO_EXEC_STEP_REQUIRED}"]})

    def expected_delta_ms(self, spec: AdapterSpec, device: DeviceProfile) -> float:
        """One DiT forward per cycle. MEASURED on H100 (probe_action_terminal_elision.py, ABBA): 22.9 ms on
        the shipped chain at 2V/4A@w5, 23.6 at 2V/2A@w1, ~35 on the stock allocator. Thor: 68.4 ms is the
        per-forward slope of the h2 design's two-point fit (section 1.3) -- PROJECTED, not measured.
        Any other device is unmeasured and gets the H100 figure."""
        if device is not None and device.capability == (11, 0):
            return 68.4
        return 22.9

    # ---- install ---------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> None:
        import modules.model as M

        Model = M.WanTransformer3DModel
        if getattr(Model, "_iwm_ate_installed", False):
            return
        check_vendor_invariants(server_module)

        _orig_forward = Model.forward
        _orig_clear_pred = Model.clear_pred_cache
        _orig_clear_cache = Model.clear_cache
        _orig_create = Model.create_empty_cache

        _ctl = controller_of

        def forward(self, input_dict, update_cache=0, cache_name="pos", action_mode=False,
                    train_mode=False):
            ctl = _ctl(self)
            if train_mode or not (ctl.armed or ctl.pending):
                return _orig_forward(self, input_dict, update_cache=update_cache,
                                     cache_name=cache_name, action_mode=action_mode,
                                     train_mode=train_mode)

            def run(*a, **kw):
                if a or kw:
                    return _orig_forward(self, *a, **kw)
                return _orig_forward(self, input_dict, update_cache=update_cache,
                                     cache_name=cache_name, action_mode=action_mode,
                                     train_mode=train_mode)

            def stash():
                # CLONE. `noisy_latents` aliases `_infer`'s `actions` when CFG is off, and `_infer`
                # keeps mutating `actions` in place after the loop (:561, :563). A materialized
                # replay must see the tensor as it was when the forward was skipped.
                import torch
                frozen = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                          for k, v in input_dict.items()}
                return ((frozen,), dict(update_cache=update_cache, cache_name=cache_name,
                                        action_mode=action_mode, train_mode=train_mode))

            return ctl.on_forward(update_cache, action_mode, run, stash)

        def clear_pred_cache(self, cache_name):
            ctl = _ctl(self)
            model = _unwrap(self)

            def reserve(args, kwargs):
                n = action_tokens(args[0])
                cn = kwargs.get("cache_name", cache_name)
                variants = {reserve_provisional_slots(b.attn1, cn, n, kwargs.get("update_cache", 1))
                            for b in model.blocks}
                variants.discard("none")
                if len(variants) != 1:
                    raise RuntimeError(
                        f"action_terminal_forward_elision: layers disagree on the allocator variant "
                        f"({sorted(variants)}); the pool is in a state no single model describes.")
                return variants.pop()

            ctl.on_clear_pred_cache(reserve)
            return _orig_clear_pred(self, cache_name)

        def clear_cache(self, cache_name):
            _ctl(self).on_cache_dropped()
            return _orig_clear_cache(self, cache_name)

        def create_empty_cache(self, *a, **kw):
            _ctl(self).on_cache_dropped()
            return _orig_create(self, *a, **kw)

        Model.forward = forward
        Model.clear_pred_cache = clear_pred_cache
        Model.clear_cache = clear_cache
        Model.create_empty_cache = create_empty_cache
        Model._iwm_ate_installed = True

        # Gate (2), and the arming scope: elide only inside `_infer`, only at video_exec_step == -1.
        _orig_infer = server_cls._infer

        def _infer(self, obs, frame_st_id=0):
            ctl = _ctl(self.transformer)
            step = getattr(self.job_config, VIDEO_EXEC_STEP_FIELD, None)
            if step != VIDEO_EXEC_STEP_REQUIRED:
                ctl.disable(f"{VIDEO_EXEC_STEP_FIELD}={step!r}; the asymmetry argument was traced "
                            f"only at {VIDEO_EXEC_STEP_REQUIRED} (configs/va_robotwin_cfg.py:30), "
                            f"where the video loop runs to its padded terminal step")
            ctl.armed = True
            try:
                return _orig_infer(self, obs, frame_st_id=frame_st_id)
            finally:
                ctl.armed = False

        server_cls._infer = _infer

    # ---- gates -----------------------------------------------------------------------------
    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(
            passed=(d == 0.0),
            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
            max_abs_delta=d,
            detail="the elided forward's output is discarded by construction and its K/V slots are "
                   "dropped by clear_pred_cache before any read; only its allocation side effect is "
                   "kept, so any nonzero delta means a reader of those slots exists")

    def benchmark(self, harness) -> BenchResult:
        before, after = harness.cycle_ms_before(), harness.cycle_ms_after()
        return BenchResult(passed=after < before, before_ms=before, after_ms=after,
                           detail="one DiT forward per cycle removed; the saved wall time is that "
                                  "forward's device time plus its host dispatch")
