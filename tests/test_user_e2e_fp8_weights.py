"""CPU evidence traversal through the serving wrappers, without loading models."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from benchmarks.regression.user_e2e import fp8_weights
from instinctflash.runtime.h100_fp8 import H100Loop


ROOT = Path(__file__).resolve().parents[1]


def plain(module, name, **fields):
    value = type(name, (), {"__module__": module})()
    vars(value).update(fields)
    return value


def packed_model():
    projection = torch.nn.Module()
    projection.register_buffer("weight_fp8", torch.zeros((16, 32), dtype=torch.float8_e4m3fn))
    model = torch.nn.Module()
    model.add_module("q_proj", projection)
    return model


# These are the real public loop classes. Only vendor construction is replaced
# with weight-free objects carrying the original class namespace and links.
@pytest.mark.parametrize("family,plugin,module,cls", [
    ("va", None, "instinctflash.adapters.lingbot_va", "_ControlLoop"),
    ("vla4", "lingbot_vla", "lingbot_vla_iwm.adapter", "_LingBotVLA4BLoop"),
    ("vla2", "lingbot_vla_v2", "lingbot_vla_v2_iwm.adapter", "_LingBotVLAV2Loop"),
    ("groot_n17", "groot_n17", "groot_n17_iwm.adapter", "_GR00TN17Loop"),
    ("pi05", "pi05_vla", "pi05_iwm.adapter", "_Pi05Loop"),
    ("cosmos", None, "instinctflash.runtime.cosmos_droid", "CosmosDROIDLoop"),
    ("cosmos_nano", None, "instinctflash.runtime.cosmos_droid", "CosmosDROIDLoop"),
    ("dreamzero", "dreamzero", "dreamzero_iwm.adapter", "_DreamZeroLoop"),
])
def test_all_family_model_bridges(monkeypatch, family, plugin, module, cls):
    if plugin:
        monkeypatch.syspath_prepend(str(ROOT / "examples" / plugin))
    loop_type = getattr(importlib.import_module(module), cls)
    loop = object.__new__(loop_type)
    model = packed_model()
    if family == "va":
        loop._server = plain("wan_va_server", "VA_Server", transformer=model)
        prefix = "_server.transformer"
    elif family in {"vla4", "vla2"}:
        policy = torch.nn.Module()
        policy.model = model
        suffix = "vla" if family == "vla4" else "vla_v2"
        loop._server = plain(f"deploy.lingbot_{suffix}_policy", "Server", vla=policy)
        prefix = "_server.vla._modules.model"
    elif family == "groot_n17":
        loop._policy = plain("gr00t.policy.gr00t_policy", "Gr00tPolicy", model=model)
        prefix = "_policy.model"
    elif family == "pi05":
        # PI05Policy is an nn.Module even though lerobot is not a permitted
        # namespace for arbitrary non-model objects.
        policy_type = type("PI05Policy", (torch.nn.Module,), {
            "__module__": "lerobot.policies.pi05.modeling_pi05"})
        loop._p = policy_type()
        loop._p.model = model
        prefix = "_p._modules.model"
    elif family in {"cosmos", "cosmos_nano"}:
        loop._service = plain("cosmos_framework.scripts.action_policy_server_robolab",
                              "RobolabPolicyService", model=model)
        prefix = "_service.model"
    else:
        trained = torch.nn.Module()
        trained.action_head = model
        policy = plain("groot.vla.model.n1_5.sim_policy", "GrootSimPolicy", trained_model=trained)
        loop._wrapper = plain("eval_utils.serve_dreamzero_wan22", "DreamZeroWan225BPolicy",
                              _policy=policy)
        prefix = "_wrapper._policy.trained_model._modules.action_head"
    wrapper = H100Loop(loop, {"precision": "fp8"}, executor="sm89_torch_fp8")
    expected = {"path": f"backend.inner.{prefix}._modules.q_proj._buffers.weight_fp8",
                "shape": [16, 32]}
    assert fp8_weights(wrapper) == [expected]
    # The original unwrapped Thor/native-policy loop remains discoverable too.
    assert fp8_weights(loop) == [{**expected, "path": expected["path"].replace(".inner", "", 1)}]


@pytest.mark.parametrize("namespace", ["wan_va_server", "wan_va.wan_va_server"])
def test_va_import_alias_does_not_hide_defining_namespace(monkeypatch, namespace):
    server = plain(namespace, "VA_Server", transformer=packed_model())
    alias = ModuleType("lingbot_va_import_alias")
    alias.VA_Server = type(server)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    assert type(server).__module__ == namespace
    assert fp8_weights(server)[0]["path"] == (
        "backend.transformer._modules.q_proj._buffers.weight_fp8")


def test_cycles_aliases_and_fused_frontend_containers_are_read_only():
    model = packed_model()
    weight = model.q_proj.weight_fp8
    before, version = weight.view(torch.uint8).clone(), weight._version
    root = plain("flash_rt.frontends.torch.pi05", "Frontend", weights={"blocks": [model]})
    root.self = root
    root.weights["same_tensor"] = (weight,)
    root.weights["parent"] = root.weights
    # Other wrappers can hold another reference to the same packed projection.
    root.capture = plain("lingbot_vla_iwm.static_capture", "Capture", model=model)
    assert fp8_weights(root) == [{
        "path": "backend.weights.blocks[0]._modules.q_proj._buffers.weight_fp8",
        "shape": [16, 32],
    }]
    assert weight._version == version
    assert torch.equal(weight.view(torch.uint8), before)


@pytest.mark.parametrize("namespace", [
    "unrelated", "deploy.unrelated", "wan_va_server_debug", "gr00t_unrelated.policy",
    "groot_unrelated", "cosmos_fake", "instinctflash_unrelated", "pi05_unrelated",
])
def test_unrelated_objects_and_prefix_lookalikes_are_not_followed(namespace):
    other = plain(namespace, "Object", model=packed_model())
    root = plain("instinctflash.runtime.h100_fp8", "Loop", unrelated=other)
    assert fp8_weights(root) == []


def test_no_module_globals_class_attributes_or_properties_are_read():
    model = packed_model()
    module = ModuleType("wan_va_server")
    module.model = model

    def inaccessible(self):
        raise AssertionError("evidence collection must not invoke model properties")

    cls = type("Policy", (), {"__module__": "gr00t.policy.gr00t_policy",
                             "global_model": model, "model": property(inaccessible)})
    root = plain("instinctflash.runtime.h100_fp8", "Loop", policy=cls(),
                 imported_module=module, imported_class=cls)
    assert fp8_weights(root) == []


@pytest.mark.parametrize("dtype,shape", [
    (torch.bfloat16, (16, 32)), (torch.float32, (16, 32)),
    (torch.int8, (16, 32)), (torch.float8_e5m2, (16, 32)),
    (torch.float8_e4m3fn, (32,)), (torch.float8_e4m3fn, (2, 16, 32)),
])
def test_recipe_metadata_and_other_tensor_formats_are_not_fp8_weight_proof(dtype, shape):
    model = torch.nn.Module()
    model.register_buffer("weight_fp8", torch.zeros(shape, dtype=dtype))
    model._sm89_fp8_recipe = {"precision": "fp8", "projections": [{"path": "q_proj"}]}
    assert fp8_weights(H100Loop(model, model._sm89_fp8_recipe)) == []
