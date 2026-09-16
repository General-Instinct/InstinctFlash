"""Explicit SM120 recipes using shared native-policy projection helpers.

Eligibility is not device or task qualification. SM89 receipts remain separate.
"""
from . import sm89_fp8 as shared
from .desktop_fp8 import SM120

EXECUTOR = SM120.executor
RECIPES = {family: shared.recipe_for(family, _target=SM120) for family in shared.RECIPES}


def recipe_for(family):
    return shared.recipe_for(family, _target=SM120)


def available(family=None):
    return shared.available(family, _target=SM120)


def requested(plan):
    return shared.requested(plan, _target=SM120)


def _require_requested_recipe(plan, family):
    return shared._require_requested_recipe(plan, family, _target=SM120)


def install_sm120_fp8(model, family, *, device=None, storage_device=None, include_mlp=False):
    return shared.install_sm89_fp8(model, family, device=device, storage_device=storage_device,
                                  include_mlp=include_mlp, _target=SM120)


def maybe_install_sm120_fp8(model, plan, family):
    return shared.maybe_install_sm89_fp8(model, plan, family, _target=SM120)


def validate_receipt(receipt, family):
    return shared.validate_receipt(receipt, family, _target=SM120)


def build_sm120_loop(adapter, checkpoint, plan, *, device=None, nfe=None, step_cache=None):
    return shared.build_sm89_loop(adapter, checkpoint, plan, device=device, nfe=nfe,
                                  step_cache=step_cache, _target=SM120)
