"""Explicit RTX 4090 E4M3 recipes around each family's native policy.

Only the named projection call sites are converted. Attention, sparse experts,
processors, schedules and episode state remain with the native adapter. These
recipes are numerical operating points; availability is not task qualification.
This module's recipe metadata is importable without Torch for offline planning.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.util import find_spec


EXECUTOR = "sm89_torch_fp8"


@dataclass(frozen=True)
class ProjectionRecipe:
    family: str
    attention_types: frozenset[str]
    projection_names: tuple[str, ...]
    requires_owned_builder: bool = False

    @property
    def recipe_id(self):
        return f"sm89_{self.family}_attention_qkv_v1"


RECIPES = {
    "pi05": ProjectionRecipe("pi05", frozenset({"GemmaAttention", "SiglipAttention"}),
                             ("q_proj", "k_proj", "v_proj")),
    "lingbot_vla": ProjectionRecipe(
        "lingbot_vla", frozenset({"Qwen2Attention", "Qwen2_5_VLAttention",
                                 "Qwen2_5_VLFlashAttention2", "GemmaAttention"}),
        ("q_proj", "k_proj", "v_proj")),
    "lingbot_vla_v2": ProjectionRecipe(
        "lingbot_vla_v2", frozenset({"Qwen3VLTextAttention", "Qwen2Attention"}),
        ("q_proj", "k_proj", "v_proj")),
    "groot_n17": ProjectionRecipe(
        "groot_n17", frozenset({"Qwen3VLTextAttention"}), ("q_proj", "k_proj", "v_proj")),
    "wan_va": ProjectionRecipe("wan_va", frozenset({"WanAttention"}),
                               ("to_q", "to_k", "to_v")),
    "cosmos3_policy": ProjectionRecipe(
        "cosmos3_policy", frozenset({"PackedAttentionMoT"}),
        ("q_proj", "k_proj", "v_proj", "q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen"),
        requires_owned_builder=True),
    "dreamzero": ProjectionRecipe("dreamzero", frozenset({"CausalWanSelfAttention"}),
                                  ("q", "k", "v"), requires_owned_builder=True),
}


def recipe_for(family: str) -> ProjectionRecipe:
    try:
        return RECIPES[family]
    except KeyError:
        raise ValueError(f"No SM89 FP8 recipe for backbone {family!r}") from None


def available(family=None) -> tuple[bool, str]:
    """Check dependencies and the selected device without launching a kernel."""
    if family is not None and family not in RECIPES:
        return False, f"No SM89 FP8 recipe for backbone {family!r}"
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        return False, "SM89 projection executor requires an RTX Ada SM89 CUDA device"
    if not callable(getattr(torch, "_scaled_mm", None)):
        return False, "SM89 E4M3 projections require torch._scaled_mm"
    if find_spec("triton") is None:
        return False, "SM89 E4M3 activation packing requires Triton"
    return True, "SM89 + PyTorch per-tensor E4M3 projections; device and task qualification are separate"


def requested(plan) -> bool:
    return any(r.name == "engine_offload" and r.applies
               and r.params.get("executor") == EXECUTOR
               for r in getattr(plan, "results", ()))


def _allowed_recipe_ids(family):
    recipe = recipe_for(family)
    allowed = {recipe.recipe_id}
    if family in {"cosmos3_policy", "dreamzero"}:
        allowed.add(recipe.recipe_id + "_dense_mlp")
    return allowed


def _require_requested_recipe(plan, family):
    candidates = [r for r in getattr(plan, "results", ())
                  if r.name == "engine_offload" and r.applies]
    if (len(candidates) != 1 or candidates[0].params.get("executor") != EXECUTOR
            or candidates[0].params.get("backend") != "engine"
            or candidates[0].params.get("recipe_id") not in _allowed_recipe_ids(family)):
        raise ValueError(f"SM89 {family} construction requires its explicit executor and recipe plan")


def install_sm89_fp8(model, family: str, *, device=None, storage_device=None, include_mlp=False):
    """Validate then pack a family's declared projections, optionally from CPU.

    CPU sources must already have their final native BF16 dtype. Only packed
    buffers move to CUDA, so a family loader can call this before model.cuda().
    With storage_device='cpu', packed buffers stay on CPU for module residency;
    packing then has no aggregate CUDA allocation and preserves the same recipe.
    The explicit dense-MLP extension is limited to Cosmos and DreamZero and has
    its own recipe identifier. It never converts sparse expert weight arrays.
    """
    import torch
    from .torch_fp8_linear import SM89FP8Linear

    recipe = recipe_for(family)
    if include_mlp and family not in {"cosmos3_policy", "dreamzero"}:
        raise ValueError(f"No SM89 dense-MLP recipe for {family}")
    if hasattr(model, "_sm89_fp8_recipe"):
        raise ValueError("SM89 FP8 already installed")
    targets = []

    def add(parent, name, path):
        source = getattr(parent, name, None)
        if type(source) is not torch.nn.Linear:
            raise ValueError(f"SM89 recipe requires a plain Linear at {path}.{name}")
        if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
            raise ValueError(f"Cannot replace hooked SM89 projection {path}.{name}")
        # Preserve native FP32 projection arithmetic instead of silently casting.
        if source.weight.dtype != torch.bfloat16:
            return
        if source.in_features % 16 or source.out_features % 16:
            raise ValueError(f"SM89 projection dimensions must be multiples of 16: {path}.{name}")
        targets.append((path, parent, name, source))

    for path, parent in model.named_modules():
        if type(parent).__name__ in recipe.attention_types:
            for name in recipe.projection_names:
                add(parent, name, path)
    attention_projections = len(targets)
    if not attention_projections:
        raise ValueError(f"No eligible native BF16 attention projections for SM89 {family}")
    if include_mlp:
        for path, parent in model.named_modules():
            if family == "cosmos3_policy" and type(parent).__name__ == "MoTDecoderLayer":
                for tower in ("mlp", "mlp_moe_gen"):
                    mlp = getattr(parent, tower)
                    fields = {"Qwen3VLTextMLP": ("gate_proj", "up_proj", "down_proj"),
                              "Nemotron3DenseVLMLP": ("up_proj", "down_proj")}.get(type(mlp).__name__)
                    if fields is None:
                        raise ValueError(f"Unsupported SM89 dense MoT MLP at {path}.{tower}")
                    for name in fields:
                        add(mlp, name, f"{path}.{tower}")
            elif family == "dreamzero" and type(parent).__name__ == "CausalWanAttentionBlock":
                for name in ("0", "2"):
                    add(parent.ffn, name, f"{path}.ffn")
        if len(targets) == attention_projections:
            raise ValueError(f"No eligible dense MLP projections for SM89 {family}")
    # Keep originals until all packing succeeds; allocation failures cannot leave
    # a partially converted model. CPU-loading bridges retain originals in RAM.
    storage_kw = {"storage_device": storage_device} if storage_device is not None else {}
    packed = [SM89FP8Linear(source, device=device, **storage_kw)
              for _, _, _, source in targets]
    receipt = {
        "executor": EXECUTOR, "precision": "fp8", "family": family,
        "recipe_id": recipe.recipe_id + ("_dense_mlp" if include_mlp else ""),
        "recipe": SM89FP8Linear.recipe, "capability": [8, 9],
        "scope": "BF16 attention Q/K/V" + (" and audited dense MLP" if include_mlp else " only"),
        "quality_status": "unverified", "quality_certificate": None,
        "packed_storage_device": str(storage_device or device or "source CUDA device"),
        "projections": [
            {"path": f"{path}.{name}".lstrip("."), "parent_type": type(parent).__name__,
             "in_features": source.in_features, "out_features": source.out_features,
             "source_device": str(source.weight.device)}
            for path, parent, name, source in targets],
    }
    for (_, parent, name, _), replacement in zip(targets, packed):
        setattr(parent, name, replacement)
    model._sm89_fp8_recipe = receipt
    return receipt


def maybe_install_sm89_fp8(model, plan, family):
    if not requested(plan):
        return
    _require_requested_recipe(plan, family)
    if hasattr(model, "_sm89_fp8_recipe"):
        # A family may have packed CPU weights before its final CUDA placement.
        return validate_receipt(model._sm89_fp8_recipe, family)
    return install_sm89_fp8(model, family)


def validate_receipt(receipt, family):
    if (not isinstance(receipt, dict) or receipt.get("executor") != EXECUTOR
            or receipt.get("precision") != "fp8" or receipt.get("family") != family
            or receipt.get("recipe_id") not in _allowed_recipe_ids(family)
            or not isinstance(receipt.get("projections"), list) or not receipt["projections"]):
        raise ValueError(f"SM89 {family} build did not install its declared FP8 projections")
    return receipt


def build_sm89_loop(adapter, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
    """Keep the family's observation/action processing and feedback ownership."""
    import torch
    from .h100_fp8 import H100Loop

    family = checkpoint.execution.backbone
    recipe = recipe_for(family)
    if (not torch.cuda.is_available()
            or torch.cuda.get_device_capability(device) != (8, 9)):
        raise RuntimeError("SM89 FP8 plan requires an SM89 CUDA device")
    _require_requested_recipe(plan, family)
    owned_builder = getattr(adapter, "build_sm89_fp8", None)
    if recipe.requires_owned_builder and not callable(owned_builder):
        raise RuntimeError(f"SM89 {family} requires an owned FP8 loading/residency builder")
    if family == "wan_va":
        for index, result in enumerate(plan.results):
            if result.name == "cfg_branch_elision" and result.applies:
                plan.results[index] = replace(result, applies=False,
                    reason="SM89 FP8 retains native CFG execution")
    if callable(owned_builder):
        kwargs = {"device": device, "nfe": nfe}
        if step_cache is not None:
            kwargs["step_cache"] = step_cache
        loop = owned_builder(checkpoint, plan, **kwargs)
    else:
        loop = adapter.build_in_process(checkpoint, plan, device=device, nfe=nfe)
    try:
        if callable(owned_builder):
            stats = loop.backend_stats
            if callable(stats):
                stats = stats()
            receipt = validate_receipt(stats.get("fp8_recipe"), family)
        else:
            if family == "pi05":
                model = loop._p.model
            elif family == "groot_n17":
                model = loop._policy.model
            elif family in {"lingbot_vla", "lingbot_vla_v2"}:
                model = loop._server.vla.model
            else:
                model = loop._server.transformer
            receipt = validate_receipt(getattr(model, "_sm89_fp8_recipe", None), family)
        # The owned low-memory builders may select the explicitly registered
        # dense-MLP extension. Expose the installed numerical recipe in explain().
        for index, result in enumerate(plan.results):
            if result.name == "engine_offload" and result.applies:
                plan.results[index] = replace(result, params={
                    **result.params, "recipe_id": receipt["recipe_id"],
                    "installed_projection_count": len(receipt["projections"]),
                    "installed_scope": receipt.get("scope"),
                })
        return H100Loop(loop, receipt, executor=EXECUTOR)
    except Exception:
        loop.close()
        raise
