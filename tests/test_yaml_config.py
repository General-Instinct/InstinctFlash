"""YAML optimization pipelines: strict input, deterministic resolution, and safe execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from contextlib import contextmanager

from instinctwm.adapters.lingbot_va import lingbot_va_spec
from instinctwm.config import (
    ConfigurationError,
    InstallPhase,
    OptimizationLayer,
    PassDefinition,
    PassRegistry,
    resolve_pipeline,
)
from instinctwm.planners.planner import Optimizer, Plan, Tier


CAPS = frozenset({"backbone:wan_va"})


@contextmanager
def _raises(error, match: str):
    try:
        yield
    except error as exc:
        assert re.search(match, str(exc)), str(exc)
    else:
        raise AssertionError(f"expected {error.__name__} matching {match!r}")


def _doc(*, graph=(), cache=(), kernel=(), policy=None):
    return {
        "schema_version": 1,
        "kind": "OptimizationPipeline",
        "name": "test",
        "policy": policy or {"tier_ceiling": "numeric", "unlisted": "off"},
        "layers": {
            "graph": list(graph), "cache": list(cache), "attention": [],
            "kernel": list(kernel), "hardware": [],
        },
    }


def test_presets_reproduce_the_current_served_chain_and_lifecycle():
    stock = resolve_pipeline("stock")
    assert stock.ordered == ()

    shipped = resolve_pipeline("shipped")
    assert [(x.id, x.definition.install_phase.value) for x in shipped.ordered] == [
        ("fsdp_elision", "pre_build"),
        ("allocator_churn_elision", "pre_build"),
        ("debug_dump_elision", "pre_build"),
        ("conditioning_prefill", "pre_build"),
        ("ring_kv_addressing", "pre_build"),
        ("conv_layout_ndhwc", "post_build"),
        ("generic_eager_stack", "post_reset"),
    ]
    from instinctwm.verify.released import shipped_configuration
    assert shipped_configuration() == [
        "--no-fsdp", "--no-empty-cache", "--no-debug-dump",
        "--conditioning-prefill", "--ring-kv", "--conv-layout",
    ]


def test_lingbot_checkpoint_nfe_is_reflected_in_planner_phases():
    from instinctwm.adapters.lingbot_va import LingBotVA

    spec = LingBotVA().spec_for_execution({"video": 2, "action": 4})
    assert spec.total_forwards() == 10
    assert spec.phase("video").nfe == 3 and spec.phase("video").commit_steps == frozenset({2})
    assert spec.phase("action").nfe == 5 and spec.phase("action").commit_steps == frozenset({4})


def test_yaml_uses_on_off_as_modes_and_is_strict():
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        good = tmp_path / "good.yaml"
        good.write_text("""\
schema_version: 1
kind: OptimizationPipeline
name: modes
policy: {tier_ceiling: bitexact, unlisted: off}
layers:
  graph:
    - {id: fsdp_elision, mode: on}
    - {id: debug_dump_elision, mode: off}
  cache: []
  attention: []
  kernel: []
  hardware: []
""")
        resolved = resolve_pipeline(good)
        assert [x.id for x in resolved.ordered] == ["fsdp_elision"]
        assert [(x.id, x.mode.value) for x in resolved.skipped] == [
            ("debug_dump_elision", "off")]

        duplicate = tmp_path / "duplicate.yaml"
        duplicate.write_text(good.read_text().replace("name: modes", "name: one\nname: two"))
        with _raises(ConfigurationError, "duplicate YAML key"):
            resolve_pipeline(duplicate)

    unknown = json.loads(json.dumps(_doc(graph=[{"id": "fsdp_elision", "bogus": 1}])))
    with _raises(ConfigurationError, "unknown field"):
        resolve_pipeline(unknown)
    wrong_layer = _doc(cache=[{"id": "fsdp_elision", "mode": "on"}])
    with _raises(ConfigurationError, "belongs to layer"):
        resolve_pipeline(wrong_layer)
    bad_param = _doc(kernel=[{
        "id": "conv_layout_ndhwc", "mode": "on", "params": {"prefer_bitexact": 1}}])
    with _raises(ConfigurationError, "must be bool"):
        resolve_pipeline(bad_param)


def test_dependencies_expand_transitively_but_explicit_off_fails_closed():
    graph = {"id": "graph_block_stack", "mode": "on"}
    policy = {"tier_ceiling": "bitexact", "allow_experimental": True, "unlisted": "off"}
    resolved = resolve_pipeline(_doc(graph=[graph], policy=policy))
    assert [x.id for x in resolved.ordered] == ["ring_kv_addressing", "graph_block_stack"]
    assert resolved.ordered[0].enabled_by == "graph_block_stack"

    with _raises(ConfigurationError, "explicitly off"):
        resolve_pipeline(_doc(
            graph=[graph], cache=[{"id": "ring_kv_addressing", "mode": "off"}],
            policy=policy))


def test_on_still_obeys_gates_and_required_turns_a_refusal_into_an_error():
    policy = {"tier_ceiling": "bitexact", "allow_experimental": True, "unlisted": "off"}
    on = _doc(graph=[{"id": "graph_block_stack", "mode": "on"}], policy=policy)
    plan = Optimizer(optimization_config=on).compile(lingbot_va_spec(), capabilities=CAPS)
    graph = next(x for x in plan.results if x.name == "graph_block_stack")
    assert not graph.applies and "slower" in graph.reason
    ring = next(x for x in plan.results if x.name == "ring_kv_addressing")
    assert not ring.applies and "no applied root" in ring.reason

    required = _doc(
        graph=[{"id": "graph_block_stack", "mode": "required"}], policy=policy)
    with _raises(ConfigurationError, "required pass.*refused"):
        Optimizer(optimization_config=required).compile(lingbot_va_spec(), capabilities=CAPS)

    required_cap = _doc(graph=[{"id": "fsdp_elision", "mode": "required"}])
    with _raises(ConfigurationError, "checkpoint does not declare"):
        Optimizer(optimization_config=required_cap).compile(
            lingbot_va_spec(), capabilities=frozenset())


def test_unlisted_auto_and_plan_round_trip():
    config = _doc(policy={"tier_ceiling": "numeric", "unlisted": "auto"})
    plan = Optimizer(optimization_config=config).compile(lingbot_va_spec(), capabilities=CAPS)
    assert {x.name for x in plan.applied} == {
        "fsdp_elision", "allocator_churn_elision", "debug_dump_elision",
        "conditioning_prefill", "ring_kv_addressing", "conv_layout_ndhwc",
    }
    with tempfile.TemporaryDirectory() as td:
        target = plan.write(Path(td) / "resolved-plan.json")
        restored = Plan.read(target)
    assert restored.to_dict() == plan.to_dict()
    assert restored.optimization_fingerprint == plan.optimization_fingerprint
    assert restored.tier() is Tier.NUMERIC
    tampered = plan.to_dict()
    tampered["results"][0]["applies"] = not tampered["results"][0]["applies"]
    with _raises(ValueError, "execution fingerprint mismatch"):
        Plan.from_dict(tampered)


def test_lifecycle_and_dependency_order_cannot_contradict_each_other():
    registry = PassRegistry()
    factory = lambda params: object()
    registry.register(PassDefinition(
        id="late", version="1", layer=OptimizationLayer.CACHE, factory=factory,
        install_phase=InstallPhase.POST_RESET))
    registry.register(PassDefinition(
        id="early", version="1", layer=OptimizationLayer.GRAPH, factory=factory,
        install_phase=InstallPhase.PRE_BUILD, requires=("late",)))
    config = _doc(graph=[{"id": "early", "mode": "on"}])
    with _raises(ConfigurationError, "dependency cycle"):
        resolve_pipeline(config, registry=registry)


def test_managed_worker_receives_the_resolved_plan_not_a_second_configuration():
    from instinctwm.adapters.lingbot_va import LingBotVA

    class _Execution:
        nfe = {"video": 2, "action": 4}
        extra = {}

    class _Checkpoint:
        execution = _Execution()

    plan = Optimizer(optimization_config="shipped").compile(
        lingbot_va_spec(), capabilities=CAPS)
    with tempfile.TemporaryDirectory() as td:
        old = os.environ.get("IWM_CACHE")
        os.environ["IWM_CACHE"] = td
        try:
            argv, _ = LingBotVA().worker_command(
                _Checkpoint(), plan, port=29061, python="python")
        finally:
            if old is None:
                os.environ.pop("IWM_CACHE", None)
            else:
                os.environ["IWM_CACHE"] = old
        assert "--optimization-plan" in argv
        artifact = Path(argv[argv.index("--optimization-plan") + 1])
        assert Plan.read(artifact).to_dict() == plan.to_dict()
        assert "--no-fsdp" not in argv
        assert argv[argv.index("--degrade-nfe") + 1] == "2,4"


def test_third_party_passes_are_discovered_and_must_be_namespaced():
    import importlib.metadata as metadata

    good = PassDefinition(
        id="example.fast_path", version="1.2.0", layer=OptimizationLayer.KERNEL,
        factory=lambda params: object())
    bad = PassDefinition(
        id="impersonates_builtin", version="1.0.0", layer=OptimizationLayer.GRAPH,
        factory=lambda params: object())

    class _EntryPoint:
        def __init__(self, name, value):
            self.name, self.value = name, f"tests:{name}"
            self._value = value

        def load(self):
            return self._value

    original = metadata.entry_points
    metadata.entry_points = lambda **kwargs: [
        _EntryPoint("good", good), _EntryPoint("bad", bad)]
    try:
        registry = PassRegistry()
        problems = registry.discover_plugins()
    finally:
        metadata.entry_points = original
    assert registry.names() == ("example.fast_path",)
    assert len(problems) == 1 and "must contain a package namespace" in problems[0]


if __name__ == "__main__":
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))
