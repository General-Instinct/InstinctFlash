"""Verify registered H100 projection modules after timing."""
import torch
from instinctflash.runtime.h100_fp8 import H100FP8Linear


def projection_evidence(model, recipe):
    modules = dict(model.named_modules())
    found = []
    for projection in recipe.get("projections", []):
        path = projection["path"]
        module = modules[path]
        if not isinstance(module, H100FP8Linear):
            raise AssertionError(f"{path}: recipe names an unconverted projection")
        weight = module.weight_fp8
        expected = [projection["out_features"], projection["in_features"]]
        if weight.dtype != torch.float8_e4m3fn or list(weight.shape) != expected:
            raise AssertionError(f"{path}: packed weight dtype/shape differs from the recipe")
        found.append({"path": path + ".weight_fp8", "shape": list(weight.shape)})
    if not found:
        raise AssertionError("H100 FP8 recipe contains no actual packed projection")
    return found
