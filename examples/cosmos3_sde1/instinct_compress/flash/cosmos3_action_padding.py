"""Public Flash adapter for CFG4 DROID with zero action-input padding.

This is a separate backbone from the historical raw fixed-SDE plugin. Native
checkpoint weights and FixedStepSampler arithmetic remain native; the frozen
padding helper constrains only the 56 non-action channels at every query and
transition. The declaration and portable sidecar make that policy explicit.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
from pathlib import Path

from instinct_compress.flash.cosmos3 import (
    Cosmos3FixedStepAdapter, _FixedStepLoop, sampling_contract,
)

BACKBONE = "cosmos3_policy_action_fixed_step"
PADDING_SIDECAR = "instinctcompress_action_padding.json"


def padding_sidecar():
    from instinct_compress.models.cosmos3_action_padding import zero_action_padding

    source = Path(inspect.getfile(inspect.unwrap(zero_action_padding)))
    return {"schema_version": 1, "action_padding": "zero", "raw_action_dim": 8,
            "model_action_dim": 64, "projection": "native_inputs_and_sde_transitions",
            "noise_rng": "unchanged_full_native_shape",
            "helper_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def action_sampling_contract(checkpoint, nfe=None):
    contract = sampling_contract(checkpoint, nfe)
    extra = checkpoint.execution.extra or {}
    if extra.get("action_padding") != "zero":
        raise ValueError("Action-aware DROID requires execution.action_padding=zero")
    if (checkpoint.execution.guidance.get("action") != {"mode": "cfg", "scale": 4.}
            or len(contract["sigmas"]) != 5):
        raise ValueError("This action-aware backbone requires four callbacks with literal CFG4")
    if (extra.get("action_dim"), extra.get("action_chunk_size"), extra.get("domain_name")) != (
            8, 32, "droid_lerobot"):
        raise ValueError("Action-aware padding requires the native 8D/32-action DROID contract")
    return contract


def validate_padding_sidecar(checkpoint):
    from instinct_compress.artifacts import validate_native_files

    root = Path(checkpoint.path)
    sidecar = root / PADDING_SIDECAR
    if not sidecar.is_file() or json.loads(sidecar.read_text()) != padding_sidecar():
        raise ValueError("Native checkpoint padding sidecar must match the installed projection helper")
    # A base pointer could silently load different bytes than the merged native
    # artifact whose padding sidecar was validated.
    # Released and merged Cosmos artifacts use a root index with component
    # subdirectory shards. Validate every referenced local payload, rather
    # than incorrectly requiring the shard files themselves at the root.
    validate_native_files(root, required_files=("checkpoint.json", PADDING_SIDECAR))


class PaddingExecution:
    """Install only the declared projection around each native generation call."""

    def __init__(self, model):
        from instinct_compress.models.cosmos3_action_padding import zero_action_padding

        self.model = model
        self.own = "generate_samples_from_batch" in vars(model)
        self.original = model.generate_samples_from_batch
        self.calls = 0
        self.branches = 0
        self.last_audit = None

        @functools.wraps(self.original)
        def generate(*args, **kwargs):
            if kwargs.get("guidance") != 4. or kwargs.get("num_steps") != 4:
                raise ValueError("Action-aware native generation requires explicit CFG4 and four calls")
            self.last_audit = None
            with zero_action_padding(model) as audit:
                result = self.original(*args, **kwargs)
            if len(audit.original_preparations) != 1 or len(audit.branches) != 8 or not audit.hooks_restored:
                raise RuntimeError("Action-aware native execution did not complete one preparation and eight CFG branches")
            self.last_audit = audit
            self.calls += 1
            self.branches += len(audit.branches)
            return result

        self.installed = generate
        model.generate_samples_from_batch = generate

    def restore(self):
        if self.model.generate_samples_from_batch is not self.installed:
            raise RuntimeError("Action-aware generation binding changed before close")
        if self.own:
            self.model.generate_samples_from_batch = self.original
        else:
            del self.model.generate_samples_from_batch


class Cosmos3ActionFixedStepAdapter(Cosmos3FixedStepAdapter):
    """DROID native Fixed4/CFG4 with an explicitly declared input projection."""

    def spec(self):
        spec = super().spec()
        guidance = {**spec.guidance, "action": dataclasses.replace(spec.guidance["action"], scale=4.)}
        return dataclasses.replace(spec, guidance=guidance,
            notes={**spec.notes, "backbone": BACKBONE, "action_padding": "zero",
                   "sampler": "Native fixed-step SDE with zero action-padding constraints"})

    def spec_for_checkpoint(self, checkpoint):
        action_sampling_contract(checkpoint)
        return super().spec_for_checkpoint(checkpoint)

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None, seed=None):
        contract = action_sampling_contract(checkpoint, nfe)
        validate_padding_sidecar(checkpoint)
        loop = super().build_in_process(checkpoint, plan, device=device, nfe=nfe, seed=seed)
        try:
            service = loop._native_loop._service
            if service.cfg.guidance != 4. or service.cfg.num_steps != 4:
                raise RuntimeError("Loaded native service differs from declared CFG4/four-step execution")
            padding = PaddingExecution(service.model)
            return _ActionFixedStepLoop(loop._native_loop, loop._sampler, contract, padding)
        except BaseException:
            loop.close()
            raise


class _ActionFixedStepLoop(_FixedStepLoop):
    def __init__(self, native_loop, sampler, contract, padding):
        super().__init__(native_loop, sampler, contract)
        self._padding = padding
        self._closed = False

    def backend_stats(self):
        return {**super().backend_stats(), "action_padding": "zero",
                "padding_projection_calls": self._padding.calls,
                "padding_projected_velocity_branches": self._padding.branches,
                "padding_hooks_restored": self._padding.last_audit is None
                    or self._padding.last_audit.hooks_restored}

    def declaration(self):
        return {**super().declaration(), "backbone": BACKBONE, "action_padding": "zero",
                "evidence": "Native FixedStepSampler and per-request zero-action-padding projection"}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._padding.restore()
        finally:
            super().close()
