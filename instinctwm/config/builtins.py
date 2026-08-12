"""Built-in YAML pass definitions.

Definitions are lazy: registering them must remain torch-free.  The installer imports a model-facing
module only after the planner has admitted the complete plan and the executor is ready to mutate it.
"""

from __future__ import annotations

from typing import Any, Mapping

from instinctwm.config.schema import (
    InstallPhase,
    OptimizationLayer,
    ParameterSpec,
    PassDefinition,
    PassMaturity,
)


WAN_VA = frozenset({"backbone:wan_va"})


class _StaticPlannerPass:
    def __init__(self, name: str, tier: str, applies, reason: str, expected: str = "unknown"):
        self.name, self._tier, self._applies = name, tier, applies
        self._reason, self._expected = reason, expected

    def evaluate(self, spec, deployment):
        from instinctwm.planners.planner import PassResult, Tier
        applies = self._applies(spec, deployment) if callable(self._applies) else bool(self._applies)
        tier = getattr(Tier, self._tier)
        reason = self._reason(spec, deployment) if callable(self._reason) else self._reason
        return PassResult(self.name, applies, tier, reason, expected_win=self._expected)


def _old(module: str, cls: str):
    def factory(params):
        from importlib import import_module
        return getattr(import_module(module), cls)(**dict(params))
    return factory


def _static(name, tier, applies, reason, expected="unknown"):
    return lambda params: _StaticPlannerPass(name, tier, applies, reason, expected)


def _runtime_installer(function_name: str):
    def install(server_module, server_cls, server, params):
        from instinctwm.runtime import lingbot_install
        fn = getattr(lingbot_install, function_name)
        if function_name == "install_conditioning_prefill":
            return list(fn(server_module, server_cls))
        return list(fn(server_module))
    return install


def _ring_installer(server_module, server_cls, server, params):
    from instinctwm.passes.lingbot.ring_kv import RingKVAddressing
    RingKVAddressing().install(server_module, server_cls)
    return ["ring_kv_addressing"]


def _hoist_installer(server_module, server_cls, server, params):
    from instinctwm.passes.lingbot.hoist_invariant_casts import HoistInvariantCasts
    return list(HoistInvariantCasts(**dict(params)).install(server_module, server_cls))


def _graph_installer(server_module, server_cls, server, params):
    from instinctwm.passes.lingbot.graph_capture import GraphBlockStack
    result = GraphBlockStack(**dict(params)).install(server_module, server_cls)
    return list(result or ["graph_block_stack"])


def _stable_installer(server_module, server_cls, server, params):
    from instinctwm.passes.lingbot.stable_pools import StableStatePools
    result = StableStatePools(**dict(params)).install(server_module, server_cls)
    return list(result or ["stable_state_pools"])


def _conv_installer(server_module, server_cls, server, params):
    if server is None:
        raise RuntimeError("conv_layout_ndhwc installs after model build and needs a server instance")
    from instinctwm.backends.conv.apply import install_conv_layout
    return list(install_conv_layout(server, **dict(params)))


def _persistent_graph_installer(server_module, server_cls, server, params):
    if server is None:
        raise RuntimeError("persistent_ring_graph installs after model build and needs a server")
    from instinctwm.passes.lingbot.persistent_graph import PersistentRingGraph
    result = PersistentRingGraph().install(server_module, server)
    if not result:
        raise RuntimeError("persistent_ring_graph did not find its ring plan-buffer capability")
    return list(result)


def _generic_installer(pass_id: str):
    """Install one generic rewrite on the shared LingBot surface, once after the first reset."""
    def install(server_module, server_cls, server, params):
        if server is None or not hasattr(server, "transformer"):
            raise RuntimeError(f"{pass_id} installs after reset and needs a built transformer")
        state = getattr(server, "_iwm_yaml_generic_passes", None)
        if state is None:
            from instinctwm.adapters.lingbot import LingBotSurface
            state = {"surface": LingBotSurface(server.transformer, server=server), "applied": set()}
            server._iwm_yaml_generic_passes = state
        if pass_id in state["applied"]:
            return []
        if pass_id == "stable_pools":
            from instinctwm.passes.stable_pools import StablePools
            implementation = StablePools()
        elif pass_id == "hoist_invariant":
            from instinctwm.passes.hoist_invariant import HoistInvariant
            implementation = HoistInvariant()
        elif pass_id == "promote_small_operand":
            from instinctwm.passes.promote_small_operand import PromoteSmallOperand
            implementation = PromoteSmallOperand()
        elif pass_id == "explicit_step_index":
            from instinctwm.passes.explicit_step_index import ExplicitStepIndex
            implementation = ExplicitStepIndex()
        else:  # pragma: no cover - definition-owned constant
            raise AssertionError(pass_id)
        from instinctwm.passes.interface import run_pass
        result = run_pass(implementation, state["surface"], None)
        if not result.fired:
            raise RuntimeError(
                f"generic_eager_stack: {pass_id} produced no rewrite: {result.skipped_reason}. "
                f"Refusing to claim the certified four-pass composition was installed.")
        state["applied"].add(pass_id)
        state[pass_id] = implementation
        return [str(result)]
    return install


def _generic_stack_installer(server_module, server_cls, server, params):
    out = []
    for pass_id in ("stable_pools", "hoist_invariant", "promote_small_operand",
                    "explicit_step_index"):
        out.extend(_generic_installer(pass_id)(server_module, server_cls, server, params))
    return out


def _persistent_streams(spec, deployment):
    from instinctwm.adapters.base import KVLifetime
    return any(s.lifetime in (KVLifetime.WINDOW, KVLifetime.EPISODE) for s in spec.streams)


def register_builtins(registry) -> None:
    # Layer 2 — GRAPH.  The three P001 components stay atomic in YAML because they have separate
    # applicability and installers even though the release ledger reports their combined result.
    registry.register(PassDefinition(
        id="fsdp_elision", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_old("instinctwm.passes.lingbot.substrate", "FSDPElision"),
        installer=_runtime_installer("install_fsdp_elision"),
        requires_capabilities=WAN_VA, auto_eligible=True, legacy_flags=("--no-fsdp",),
        description="Remove world-size-one FSDP wrapping."))
    registry.register(PassDefinition(
        id="allocator_churn_elision", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_old("instinctwm.passes.lingbot.substrate", "AllocatorChurnElision"),
        installer=_runtime_installer("install_allocator_churn_elision"),
        after=("fsdp_elision",),
        requires_capabilities=WAN_VA, auto_eligible=True, legacy_flags=("--no-empty-cache",)))
    registry.register(PassDefinition(
        id="debug_dump_elision", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_old("instinctwm.passes.lingbot.substrate", "DebugDumpElision"),
        installer=_runtime_installer("install_debug_dump_elision"),
        after=("allocator_churn_elision",),
        requires_capabilities=WAN_VA, auto_eligible=True, legacy_flags=("--no-debug-dump",)))
    registry.register(PassDefinition(
        id="generic_eager_stack", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_static(
            "generic_eager_stack", "BITEXACT", True,
            "the four-rewrite stack is 26.5 ms faster in sequential A/B and bit-exact; "
            "its hoist component is not exposed alone because that intermediate regresses 133.4 ms",
            "26.5 ms late-episode gain over P003 in sequential A/B"),
        installer=_generic_stack_installer, install_phase=InstallPhase.POST_RESET,
        after=("ring_kv_addressing",), requires_capabilities=WAN_VA,
        description="Certified StablePools + HoistInvariant + PromoteSmallOperand + "
                    "ExplicitStepIndex composition."))
    registry.register(PassDefinition(
        id="obs_decode_elision", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_old("instinctwm.passes.lingbot.substrate", "ObsDecodeElision"),
        requires_capabilities=WAN_VA, auto_eligible=False,
        no_runtime_action="the action-only serving path never invokes the observation decoder"))
    registry.register(PassDefinition(
        id="hoist_invariant_casts", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_static("hoist_invariant_casts", "BITEXACT", True,
                        "model constants are invariant across forwards; released bit-exact gate"),
        installer=_hoist_installer, requires_capabilities=WAN_VA, auto_eligible=False,
        legacy_flags=("--hoist-casts",)))
    registry.register(PassDefinition(
        id="graph_block_stack", version="1.0.1", layer=OptimizationLayer.GRAPH,
        factory=_static("graph_block_stack", "BITEXACT", False,
                        "current Fast operating point measures graph capture 1.43x slower; refused"),
        installer=_graph_installer, requires=("ring_kv_addressing",),
        after=("conditioning_prefill", "hoist_invariant_casts", "stable_state_pools"),
        requires_capabilities=WAN_VA, maturity=PassMaturity.EXPERIMENTAL,
        params={"verbose": ParameterSpec(bool, default=True),
                "max_graphs": ParameterSpec(int, default=64, minimum=1, maximum=4096)},
        legacy_flags=("--graph-blocks",)))
    registry.register(PassDefinition(
        id="stable_state_pools", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_static("stable_state_pools", "BITEXACT", False,
                        "only benefits graph_block_stack, which is currently refused by measurement"),
        installer=_stable_installer, requires=("ring_kv_addressing",),
        before=("graph_block_stack",), requires_capabilities=WAN_VA,
        maturity=PassMaturity.EXPERIMENTAL,
        params={"verbose": ParameterSpec(bool, default=True)}, legacy_flags=("--stable-pools",)))
    registry.register(PassDefinition(
        id="persistent_ring_graph", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=_static(
            "persistent_ring_graph", "BITEXACT", False,
            "paired episode-mode measurement is 503.5 ms versus 351.4 ms with capture off; refused"),
        installer=_persistent_graph_installer, install_phase=InstallPhase.POST_BUILD,
        requires=("graph_block_stack", "ring_kv_addressing"),
        requires_capabilities=WAN_VA, maturity=PassMaturity.EXPERIMENTAL,
        legacy_flags=("--persistent-graph",)))

    # Layer 3 — CACHE.
    registry.register(PassDefinition(
        id="conditioning_prefill", version="1.0.0", layer=OptimizationLayer.CACHE,
        factory=_old("instinctwm.passes.lingbot.conditioning_prefill", "ConditioningPrefill"),
        installer=_runtime_installer("install_conditioning_prefill"),
        after=("debug_dump_elision",),
        requires_capabilities=WAN_VA, auto_eligible=True,
        legacy_flags=("--conditioning-prefill",)))
    registry.register(PassDefinition(
        id="ring_kv_addressing", version="1.0.0", layer=OptimizationLayer.CACHE,
        factory=_static(
            "ring_kv_addressing", "BITEXACT", _persistent_streams,
            lambda spec, deployment: (
                "persistent KV streams can use interval addressing"
                if _persistent_streams(spec, deployment)
                else "no persistent KV stream; there is no pool to re-address"),
            "released 1.40x step speedup with wraparound parity"),
        installer=_ring_installer, requires_capabilities=WAN_VA, auto_eligible=True,
        after=("conditioning_prefill",),
        legacy_flags=("--ring-kv",)))

    # Layer 5 — KERNEL. Selection is declarative; conversion happens post-build when VAEs exist.
    registry.register(PassDefinition(
        id="conv_layout_ndhwc", version="1.0.0", layer=OptimizationLayer.KERNEL,
        factory=_static(
            "conv_layout_ndhwc", "NUMERIC", True,
            "measured cuDNN NDHWC plan is profitable and carries a paired non-inferiority certificate",
            "1.405x at the current operating point"),
        installer=_conv_installer, install_phase=InstallPhase.POST_BUILD,
        after=("ring_kv_addressing",),
        params={"prefer_bitexact": ParameterSpec(bool, default=False)},
        requires_capabilities=WAN_VA, auto_eligible=True, legacy_flags=("--conv-layout",)))
