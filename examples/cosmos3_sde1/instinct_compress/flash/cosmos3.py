"""Serve native Cosmos DROID checkpoints with an explicit fixed sigma schedule.

This optional plugin depends on InstinctFlash's Cosmos3 adapter and NVIDIA's
Cosmos Framework. The ordinary DROID service otherwise defaults to UniPC even
when a checkpoint contains fixed-step sampler configuration. Selecting this
backbone therefore carries an actual execution difference, independent of the
training method that produced its weights.
"""

from __future__ import annotations

import dataclasses
import functools
import math

from cosmos3_iwm.adapter import Cosmos3PolicyAdapter

BACKBONE = "cosmos3_policy_fixed_step"


def sampling_contract(checkpoint, nfe=None):
    extra = checkpoint.execution.extra or {}
    spec = extra.get("sampling")
    if not isinstance(spec, dict) or spec.get("kind") != "rectified_flow_fixed_step":
        raise ValueError("This adapter requires execution.sampling.kind=rectified_flow_fixed_step")
    if spec.get("sample_type") != "sde":
        raise ValueError("The native Cosmos FixedStepSampler requires sample_type=sde")
    sigmas = spec.get("sigmas")
    if not isinstance(sigmas, list) or len(sigmas) < 2:
        raise ValueError("Fixed-step sigmas must include noise start 1 and terminal 0")
    if any(isinstance(s, bool) or not isinstance(s, (float, int)) or not math.isfinite(s) for s in sigmas):
        raise ValueError("Fixed-step sigmas must be finite numbers")
    if sigmas[0] != 1.0 or sigmas[-1] != 0.0 or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("Fixed-step sigmas must decrease strictly from 1 to 0")
    steps = len(sigmas) - 1
    declared = dict(checkpoint.execution.nfe or {})
    effective = {**declared, **dict(nfe or {})}
    if declared.get("action") != steps or effective.get("action") != steps:
        raise ValueError("Action NFE must match the complete trained fixed-step schedule; truncation is not supported")
    if effective.get("prefix", 1) != 1:
        raise ValueError("Cosmos DROID requires exactly one prefix phase")
    scale = spec.get("num_train_timesteps")
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Declare a positive num_train_timesteps for the native time embedding")
    return {"kind": spec["kind"], "sigmas": [float(s) for s in sigmas],
            "sample_type": "sde", "num_train_timesteps": float(scale)}


class Cosmos3FixedStepAdapter(Cosmos3PolicyAdapter):
    """The native DROID adapter with a declared fixed-step sampling capability."""

    def spec(self):
        spec = super().spec()
        phases = tuple(dataclasses.replace(p, truncatable=False) if p.name == "action" else p
                       for p in spec.phases)
        return dataclasses.replace(spec, phases=phases,
                                   notes={**spec.notes, "backbone": BACKBONE,
                                          "sampler": "Explicit rectified-flow fixed-step SDE grid"})

    def spec_for_checkpoint(self, checkpoint):
        contract = sampling_contract(checkpoint)
        spec = self.spec()
        phases = tuple(dataclasses.replace(p, nfe=len(contract["sigmas"]) - 1) if p.name == "action" else p
                       for p in spec.phases)
        return dataclasses.replace(spec, phases=phases)

    def build_in_process(self, checkpoint, plan, *, device=None, nfe=None, seed=None):
        contract = sampling_contract(checkpoint, nfe)
        if seed is not None:
            execution = dataclasses.replace(checkpoint.execution,
                                            extra={**dict(checkpoint.execution.extra or {}), "seed": int(seed)})
            checkpoint = dataclasses.replace(checkpoint, execution=execution)
        # The family adapter owns camera assembly, prompts, normalization and
        # action decoding. Its model exposes a public per-call sampler argument.
        loop = super().build_in_process(checkpoint, plan, device=device, nfe=nfe)
        try:
            sampler = _install_sampler(loop._service.model, contract)
            return _FixedStepLoop(loop, sampler, contract)
        except Exception:
            loop.close()
            raise

    def build_fp8(self, checkpoint, *, device=None, nfe=None):
        raise ValueError("Fixed-step Cosmos checkpoints currently support native precision only")


def _install_sampler(model, contract):
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler

    expected = contract["num_train_timesteps"]
    actual = float(model.config.rectified_flow_inference_config.num_train_timesteps)
    if actual != expected:
        raise ValueError(f"Native timestep scale {actual} disagrees with declaration {expected}")
    config = getattr(model.config, "fixed_step_sampler_config", None)
    if config is None:
        raise ValueError("Native Cosmos config must carry fixed_step_sampler_config alongside the serving declaration")
    grid = list(config.t_list)
    if grid and grid[-1] != 0.0:
        grid.append(0.0)
    if grid != contract["sigmas"] or config.sample_type != contract["sample_type"]:
        raise ValueError("Native fixed-step sampler config disagrees with the serving declaration")

    class MeasuredFixedStepSampler(FixedStepSampler):
        def __init__(self):
            super().__init__(t_list=contract["sigmas"], sample_type=contract["sample_type"],
                             num_train_timesteps=expected)
            self.calls = 0
            self.velocity_evaluations = 0

        def __call__(self, velocity_fn, *args, **kwargs):
            def measured(*args, **kwargs):
                self.velocity_evaluations += 1
                return velocity_fn(*args, **kwargs)

            result = super().__call__(measured, *args, **kwargs)
            self.calls += 1
            return result

    sampler = MeasuredFixedStepSampler()
    native_generate = model.generate_samples_from_batch

    @functools.wraps(native_generate)
    def generate(*args, **kwargs):
        if "sampler" in kwargs and kwargs["sampler"] is not sampler:
            raise ValueError("Cannot override a checkpoint's declared fixed-step sampler")
        kwargs["sampler"] = sampler
        return native_generate(*args, **kwargs)

    model.generate_samples_from_batch = generate
    return sampler


class _FixedStepLoop:
    def __init__(self, native_loop, sampler, contract):
        self._native_loop, self._sampler, self._contract = native_loop, sampler, contract

    def predict(self, observation, *, executed_action=None):
        return self._native_loop.predict(observation, executed_action=executed_action)

    def reset(self, **conditioning):
        self._native_loop.reset(**conditioning)

    def backend_stats(self):
        return {**self._native_loop.backend_stats(), "sampling": self._contract,
                "sampler_calls": self._sampler.calls,
                "velocity_evaluations": self._sampler.velocity_evaluations}

    def declaration(self):
        return {**self._native_loop.declaration(), "backbone": BACKBONE,
                "sampling": self._contract,
                "evidence": "NVIDIA FixedStepSampler passed explicitly to native generation"}

    def close(self):
        self._native_loop.close()
