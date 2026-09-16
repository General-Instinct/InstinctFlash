"""CPU contract checks for explicit SM89 dispatch; GPU arithmetic is separate."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from instinctflash.descriptors.deployment import DeploymentSpec
from instinctflash.passes.contract import DeviceProfile
from instinctflash.passes.generic.engine_offload import EngineOffloadApplicable
from instinctflash.planners.planner import PassResult, Plan, Tier
from instinctflash.runtime import sm89_fp8
from instinctflash.runtime.execution import _mark_plan_engine_executed, choose_backend
from instinctflash.runtime.precision import install_requested_fp8
from instinctflash.runtime.torch_fp8_linear import SM89FP8Linear, ThorFP8Linear


class GemmaAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        self.o_proj = torch.nn.Linear(16, 16, dtype=torch.bfloat16)


class FakePacked(torch.nn.Identity):
    recipe = "test-only-packed-projection"

    def __init__(self, source, *, device=None):
        super().__init__()


def make_plan(family="pi05"):
    return Plan("test", [PassResult(
        "engine_offload", True, Tier.NUMERIC, "test candidate",
        params={"backend": "engine", "executor": sm89_fp8.EXECUTOR,
                "recipe_id": sm89_fp8.recipe_for(family).recipe_id})])


def receipt(family="pi05"):
    return {"executor": sm89_fp8.EXECUTOR, "precision": "fp8", "family": family,
            "recipe_id": sm89_fp8.recipe_for(family).recipe_id,
            "projections": [{"path": "attention.q_proj"}]}


class SM89Contracts(unittest.TestCase):
    def test_cpu_weight_scale_keeps_cuda_host_scalar_rounding(self):
        source = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            source.weight.fill_(0.0361328125)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            packed = SM89FP8Linear(source, device="cuda:0", storage_device="cpu")
        # Actual SM89 CUDA scale is 0x38a92493; CPU division returns 0x38a92492.
        self.assertEqual(packed.weight_scale.view(torch.int32).item(), 0x38A92493)
        self.assertTrue(all(value.device.type == "cpu" for value in packed.buffers()))

    def test_planner_selects_explicit_registered_recipe_on_sm89(self):
        device = DeviceProfile(name="RTX 4090", capability=(8, 9), total_memory=24 << 30,
                               features=frozenset({"cuda", "fp8"}))
        for family in sm89_fp8.RECIPES:
            result = EngineOffloadApplicable().evaluate(
                SimpleNamespace(notes={"backbone": family}), DeploymentSpec(device=device))
            self.assertTrue(result.applies, family)
            self.assertEqual(result.params["executor"], sm89_fp8.EXECUTOR)
            self.assertEqual(result.tier, Tier.NUMERIC)
            self.assertIn("qualification", result.reason)
        result = EngineOffloadApplicable().evaluate(
            SimpleNamespace(notes={"backbone": "unknown"}), DeploymentSpec(device=device))
        self.assertFalse(result.applies)
        self.assertNotIn("backend", result.params)

    def test_h100_keeps_its_executor(self):
        device = DeviceProfile(name="H100", capability=(9, 0), total_memory=80 << 30,
                               features=frozenset({"cuda", "fp8"}))
        result = EngineOffloadApplicable().evaluate(
            SimpleNamespace(notes={"backbone": "pi05"}), DeploymentSpec(device=device))
        self.assertEqual(result.params["executor"], "h100_torch_fp8")

    def test_native_install_does_not_convert_and_explicit_plan_does(self):
        model = GemmaAttention()
        with patch.object(sm89_fp8, "maybe_install_sm89_fp8") as install:
            install_requested_fp8(model, SimpleNamespace(results=[]), "pi05")
            install.assert_not_called()
            plan = make_plan()
            install_requested_fp8(model, plan, "pi05")
            install.assert_called_once_with(model, plan, "pi05")

    def test_runtime_reports_native_passes_for_projection_executor(self):
        plan = make_plan()
        native = PassResult("graph_capture", True, Tier.BITEXACT, "native graph")
        plan.results.append(native)
        _mark_plan_engine_executed(plan, plan.results[0])
        self.assertIs(plan.results[1], native)
        self.assertTrue(native.applies)

    def test_requested_fp8_passes_family_to_sm89_dependency_probe(self):
        checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="pi05"))
        with patch("instinctflash.runtime.engine_backend.engine_available",
                   return_value=(True, "SM89")) as probe, \
             patch("instinctflash.runtime.engine_backend.EngineBackend") as backend:
            result, _ = choose_backend("auto", object(), checkpoint, make_plan(), precision="fp8")
        probe.assert_called_once_with("pi05")
        self.assertIs(result, backend.return_value)

    def test_family_recipe_does_not_replace_outputs_or_unrelated_linears(self):
        model = torch.nn.Sequential(GemmaAttention(), torch.nn.Linear(16, 16))
        output, unrelated = model[0].o_proj, model[1]
        with patch("instinctflash.runtime.torch_fp8_linear.SM89FP8Linear", FakePacked):
            result = sm89_fp8.install_sm89_fp8(model, "pi05", device="cuda:2")
        self.assertEqual(len(result["projections"]), 3)
        self.assertIs(model[0].o_proj, output)
        self.assertIs(model[1], unrelated)
        self.assertEqual(result["quality_status"], "unverified")
        self.assertIsNone(result["quality_certificate"])
        self.assertEqual({item["source_device"] for item in result["projections"]}, {"cpu"})
        with self.assertRaisesRegex(ValueError, "already installed"):
            sm89_fp8.install_sm89_fp8(model, "pi05")

    def test_failed_packing_leaves_every_original_projection(self):
        model = torch.nn.Sequential(GemmaAttention(), GemmaAttention())
        originals = dict(model.named_modules())
        calls = []
        def fail_later(source, **kwargs):
            calls.append(source)
            if len(calls) == 5:
                raise RuntimeError("allocation failed")
            return FakePacked(source)
        with patch("instinctflash.runtime.torch_fp8_linear.SM89FP8Linear", fail_later):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                sm89_fp8.install_sm89_fp8(model, "pi05")
        self.assertEqual(dict(model.named_modules()), originals)
        self.assertFalse(hasattr(model, "_sm89_fp8_recipe"))

    def test_hooked_and_custom_linear_are_refused_before_conversion(self):
        model = GemmaAttention()
        model.v_proj.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "hooked"):
            sm89_fp8.install_sm89_fp8(model, "pi05")
        class CustomLinear(torch.nn.Linear):
            def forward(self, x):
                return super().forward(x) + 1
        model = GemmaAttention()
        model.q_proj = CustomLinear(16, 16, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "plain Linear"):
            sm89_fp8.install_sm89_fp8(model, "pi05")

    def test_fp32_weights_are_preserved_without_silent_bf16_conversion(self):
        model = GemmaAttention().float()
        with self.assertRaisesRegex(ValueError, "No eligible native BF16"):
            sm89_fp8.install_sm89_fp8(model, "pi05")
        self.assertEqual(model.q_proj.weight.dtype, torch.float32)

    def test_attention_and_dense_mlp_extension_have_distinct_receipts(self):
        class CausalWanSelfAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
                self.k = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
                self.v = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
                self.o = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        class CausalWanAttentionBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = CausalWanSelfAttention()
                self.ffn = torch.nn.Sequential(
                    torch.nn.Linear(16, 32, dtype=torch.bfloat16), torch.nn.GELU(),
                    torch.nn.Linear(32, 16, dtype=torch.bfloat16))
        attention_only, dense = CausalWanAttentionBlock(), CausalWanAttentionBlock()
        untouched = attention_only.ffn[0]
        with patch("instinctflash.runtime.torch_fp8_linear.SM89FP8Linear", FakePacked):
            base = sm89_fp8.install_sm89_fp8(attention_only, "dreamzero")
            extended = sm89_fp8.install_sm89_fp8(dense, "dreamzero", include_mlp=True)
        self.assertIs(attention_only.ffn[0], untouched)
        self.assertEqual(len(base["projections"]), 3)
        self.assertEqual(len(extended["projections"]), 5)
        self.assertEqual(extended["recipe_id"], base["recipe_id"] + "_dense_mlp")
        self.assertIsInstance(dense.attention.o, torch.nn.Linear)
        self.assertIsInstance(dense.ffn[1], torch.nn.GELU)
        with self.assertRaisesRegex(ValueError, "No SM89 dense-MLP"):
            sm89_fp8.install_sm89_fp8(GemmaAttention(), "pi05", include_mlp=True)

    def test_recipe_from_another_family_is_refused_before_packing(self):
        with patch("instinctflash.runtime.torch_fp8_linear.SM89FP8Linear") as pack:
            with self.assertRaisesRegex(ValueError, "explicit executor and recipe"):
                sm89_fp8.maybe_install_sm89_fp8(GemmaAttention(), make_plan("wan_va"), "pi05")
        pack.assert_not_called()

    def test_receipt_requires_correct_family_executor_and_nonempty_projections(self):
        good = receipt()
        self.assertIs(sm89_fp8.validate_receipt(good, "pi05"), good)
        for field, value in (("family", "groot_n17"), ("executor", "h100_torch_fp8"),
                             ("recipe_id", "unknown"), ("projections", [])):
            with self.assertRaises(ValueError):
                sm89_fp8.validate_receipt({**good, field: value}, "pi05")

    def test_missing_memory_builder_refuses_before_model_loading(self):
        checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="dreamzero"))
        adapter = SimpleNamespace(build_in_process=lambda *a, **k: self.fail("weights loaded"))
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            with self.assertRaisesRegex(RuntimeError, "owned FP8 loading/residency"):
                sm89_fp8.build_sm89_loop(adapter, checkpoint, make_plan("dreamzero"))

    def test_owned_builder_failure_closes_native_loop(self):
        checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="cosmos3_policy"))
        from unittest.mock import Mock
        loop = SimpleNamespace(backend_stats={"fp8_recipe": {}}, close=Mock())
        adapter = SimpleNamespace(build_sm89_fp8=Mock(return_value=loop))
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            with self.assertRaisesRegex(ValueError, "declared FP8 projections"):
                sm89_fp8.build_sm89_loop(adapter, checkpoint, make_plan("cosmos3_policy"))
        loop.close.assert_called_once()

    def test_owned_dense_recipe_is_reported_in_the_executed_plan(self):
        checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="cosmos3_policy"))
        from unittest.mock import Mock
        installed = receipt("cosmos3_policy")
        installed["recipe_id"] += "_dense_mlp"
        installed["scope"] = "attention QKV and dense MLP"
        loop = SimpleNamespace(backend_stats={"fp8_recipe": installed}, close=Mock())
        adapter = SimpleNamespace(build_sm89_fp8=Mock(return_value=loop))
        plan = make_plan("cosmos3_policy")
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            wrapped = sm89_fp8.build_sm89_loop(adapter, checkpoint, plan)
        self.assertEqual(wrapped.declaration()["executor"], sm89_fp8.EXECUTOR)
        self.assertEqual(plan.results[0].params["recipe_id"], installed["recipe_id"])
        self.assertEqual(plan.results[0].params["installed_scope"], installed["scope"])
        loop.close.assert_not_called()

    def test_projection_classes_keep_architecture_contracts_separate(self):
        linear = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            with self.assertRaisesRegex(ValueError, "capability"):
                ThorFP8Linear(linear, device="cuda")
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaisesRegex(ValueError, "capability"):
                SM89FP8Linear(linear, device="cuda")

    def test_cpu_packed_storage_preserves_recipe_without_cuda_tensor_allocation(self):
        linear = torch.nn.Linear(32, 16, dtype=torch.bfloat16)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            packed = SM89FP8Linear(linear, device="cuda", storage_device="cpu")
        expected_scale = linear.weight.float().abs().amax().clamp_min(1e-12).reshape(1) / 448.
        expected = (linear.weight.float() / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(packed.weight_scale, expected_scale))
        self.assertTrue(torch.equal(packed.weight_fp8.view(torch.uint8), expected.view(torch.uint8)))
        self.assertTrue(all(t.device.type == "cpu" for t in packed.buffers()))
        self.assertTrue(torch.equal(packed.bias, linear.bias))

    def test_family_installer_can_pack_all_cpu_without_cuda_resident_weights(self):
        model = GemmaAttention()
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            result = sm89_fp8.install_sm89_fp8(model, "pi05", device="cuda", storage_device="cpu")
        self.assertEqual(result["packed_storage_device"], "cpu")
        self.assertIsInstance(model.q_proj, SM89FP8Linear)
        self.assertTrue(all(t.device.type == "cpu" for t in model.buffers()))


if __name__ == "__main__":
    unittest.main()
