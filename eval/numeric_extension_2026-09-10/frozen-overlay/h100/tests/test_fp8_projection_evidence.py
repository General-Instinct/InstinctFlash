"""The benchmark must inspect real FP8 modules rather than trust a recipe list."""
import importlib.util
from pathlib import Path

import pytest
import torch
from instinctflash.runtime.h100_fp8 import H100FP8Linear

path = Path(__file__).resolve().parents[1] / 'eval/numeric_extension_2026-09-10/fp8_evidence.py'
spec = importlib.util.spec_from_file_location('fp8_projection_evidence', path)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
RECIPE = {'projections': [{'path': 'q', 'in_features': 32, 'out_features': 16}]}


def model_with_buffer(dtype):
    # Allocation-only fixture: no CUDA conversion or inference is simulated.
    module = H100FP8Linear.__new__(H100FP8Linear)
    torch.nn.Module.__init__(module)
    module.register_buffer('weight_fp8', torch.empty((16, 32), dtype=dtype))
    model = torch.nn.Module()
    model.add_module('q', module)
    return model


def test_registered_h100_fp8_buffers_match_recipe():
    result = evidence.projection_evidence(model_with_buffer(torch.float8_e4m3fn), RECIPE)
    assert result == [{'path': 'q.weight_fp8', 'shape': [16, 32]}]


def test_unconverted_projection_cannot_pass_by_recipe_alone():
    model = torch.nn.Module()
    model.add_module('q', torch.nn.Linear(32, 16))
    with pytest.raises(AssertionError, match='unconverted'):
        evidence.projection_evidence(model, RECIPE)


def test_wrong_dtype_or_empty_recipe_is_rejected():
    with pytest.raises(AssertionError, match='dtype/shape'):
        evidence.projection_evidence(model_with_buffer(torch.bfloat16), RECIPE)
    with pytest.raises(AssertionError, match='no actual'):
        evidence.projection_evidence(torch.nn.Module(), {'projections': []})
