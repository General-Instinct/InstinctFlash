"""Opt-in DROID action-padding ablation around the original native sampler.

The released RF training path zeros padded action inputs after interpolation.
The original FixedStepSampler re-noises them. This context preserves the native
sampler's RNG, arithmetic and number of calls, adding only zero-valued padding
constraints to its existing conditioning mask. It is a separate execution
contract, not a change to the frozen raw-SDE experiment or its checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import math

import torch


@dataclass(frozen=True)
class ActionLayout:
    offset: int
    shape: tuple[int, int]
    raw_dim: int = 8

    def view(self, value):
        if value.ndim != 1 or value.numel() != self.offset + self.shape[0] * self.shape[1]:
            raise ValueError("Expected the native flat [vision | DROID action] layout")
        return value[self.offset:].reshape(self.shape)

    def project(self, value, fill=0):
        result = value.clone()
        self.view(result)[:, self.raw_dim:] = fill
        return result

    def valid(self, value):
        return torch.cat((value[:self.offset], self.view(value)[:, :self.raw_dim].flatten()))


def action_layouts(sequence_plans, gen_data_clean):
    """Resolve layout from native metadata; reject ambiguous/non-DROID inputs."""
    actions = gen_data_clean.x0_tokens_action
    raw_dims = gen_data_clean.raw_action_dim
    vision = gen_data_clean.x0_tokens_vision
    counts = gen_data_clean.num_vision_items_per_sample
    if (not sequence_plans or actions is None or raw_dims is None
            or len(actions) != len(sequence_plans) or len(raw_dims) != len(sequence_plans)):
        raise ValueError("Every sample must have explicit DROID action padding metadata")
    if counts is not None and len(counts) != len(sequence_plans):
        raise ValueError("Native vision item counts do not match the sample batch")
    result, vision_index = [], 0
    for index, plan in enumerate(sequence_plans):
        if not plan.has_action or plan.has_sound:
            raise ValueError("Padding ablation supports DROID vision/action samples without sound")
        shape = tuple(actions[index].shape)
        raw_dim = raw_dims[index]
        if raw_dim is None or int(raw_dim) != 8 or len(shape) != 2 or shape[1] != 64:
            raise ValueError("Padding ablation requires explicit raw action width 8 and model width 64")
        count = int(counts[index]) if counts is not None else 1
        if count < 1 or vision_index + count > len(vision):
            raise ValueError("Invalid native vision layout")
        offset = sum(item.numel() for item in vision[vision_index:vision_index + count])
        result.append(ActionLayout(offset, shape))
        vision_index += count
    if vision_index != len(vision):
        raise ValueError("Unassigned native vision items")
    return result


def _condition_layout(condition):
    shape = tuple(condition.action_shape)
    if len(shape) != 2 or shape[1] != 64:
        raise ValueError("Expected DROID model action width 64")
    if condition.reference.ndim != 2:
        raise ValueError("Expected batched flat native condition references")
    layout = ActionLayout(int(condition.action_offset), shape)
    layout.view(condition.reference[0])
    native_layouts = action_layouts(condition.sequence_plans, condition.gen_data_clean)
    if len(native_layouts) != len(condition.reference) or any(item != layout for item in native_layouts):
        raise ValueError("Prepared condition layout differs from native DROID metadata")
    return layout


def project_action_padding(latent, condition):
    """Differentiable P(x): zero only action columns 8:64, retaining all else."""
    layout = _condition_layout(condition)
    if latent.shape != condition.reference.shape:
        raise ValueError("Native projected latent shape differs from its condition")
    return torch.stack([layout.project(value) for value in latent.unbind(0)])


def action_condition_mask(condition):
    """Native frame mask plus deterministic zero-padding constraints."""
    from instinct_compress.models.cosmos3_distill_rollout import native_condition_mask

    layout = _condition_layout(condition)
    mask = native_condition_mask(condition)
    return torch.stack([layout.project(value, 1) for value in mask.unbind(0)])


def native_action_velocity(adapter, latent, time, condition, *, guidance=None):
    """Actual native BF16 guided velocity evaluated on P(x)."""
    from instinct_compress.models.cosmos3_distill_rollout import native_velocity

    return native_velocity(adapter, project_action_padding(latent, condition), time,
                           condition, guidance=guidance)


def native_action_clean(adapter, latent, time, condition, *, guidance=None):
    """FP32 RF score conversion after native BF16 velocity/CFG on P(x)."""
    state = project_action_padding(latent, condition)
    velocity = native_action_velocity(adapter, state, time, condition, guidance=guidance)
    clean = state.float() - time.reshape(-1, 1).float() * velocity.float()
    mask = action_condition_mask(condition).to(clean)
    reference = project_action_padding(condition.reference, condition).to(clean)
    return mask * reference + (1 - mask) * clean


def renoise_action_generated(generated, condition, time, noise):
    """Full native RF noise draw, with deterministic action pads projected out."""
    from instinct_compress.models.cosmos3_distill_rollout import renoise_generated

    return project_action_padding(renoise_generated(generated, condition, time, noise), condition)


class ActionAwareFixedStepSampler:
    """Stock native FixedStepSampler with explicit zero-valued action padding.

    Native SDE draws all coordinates before applying the augmented mask. This
    retains the exact RNG stream for every real action and vision coordinate.
    The wrapped velocity also projects inputs defensively, including the first
    call if the supplied initial state contains padded noise.
    """

    def __init__(self, condition, *, t_list=(1., .75, .5, .25, 0.),
                 num_train_timesteps=1000.):
        from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler

        levels = tuple(float(value) for value in t_list)
        if (not levels or levels[0] != 1. or not all(math.isfinite(v) for v in levels)
                or any(a <= b for a, b in zip(levels, levels[1:])) or levels[-1] < 0):
            raise ValueError("Expected descending native RF levels from 1 toward 0")
        self.condition = condition
        self.layout = _condition_layout(condition)
        self.native = FixedStepSampler(list(levels), sample_type="sde",
                                       num_train_timesteps=num_train_timesteps)
        self.t_list = self.native.t_list
        self.sample_type = self.native.sample_type
        self.num_train_timesteps = self.native.num_train_timesteps
        self.callback_padding = []

    def __call__(self, velocity_fn, noise, num_steps=None, shift=None, seed=None,
                 condition_reference=None, condition_mask=None):
        if condition_reference is None or condition_mask is None:
            raise ValueError("Action-aware native sampling requires explicit conditioning")
        is_list = isinstance(noise, list)
        if is_list != isinstance(condition_reference, list) or is_list != isinstance(condition_mask, list):
            raise ValueError("Native noise/reference/mask container kinds differ")
        values = noise if is_list else [noise]
        references = condition_reference if is_list else [condition_reference]
        masks = condition_mask if is_list else [condition_mask]
        if len(values) != len(references) or len(values) != len(masks):
            raise ValueError("Native noise/reference/mask batch sizes differ")
        projected = [self.layout.project(value) for value in values]
        refs = [self.layout.project(value) for value in references]
        augmented = [self.layout.project(value, 1) for value in masks]
        self.callback_padding = []

        def callback(state, timestep):
            items = state if isinstance(state, list) else [state]
            self.callback_padding.append([self.layout.view(value)[:, 8:].detach().clone() for value in items])
            result = [self.layout.project(value) for value in items]
            return velocity_fn(result if is_list else result[0], timestep)

        return self.native(callback, projected if is_list else projected[0],
            num_steps=num_steps, shift=shift, seed=seed,
            condition_reference=refs if is_list else refs[0],
            condition_mask=augmented if is_list else augmented[0])


def rollout_action_fixed(adapter, condition, *, times=(1., .75, .5, .25, 0.),
                         prefix_steps=4, grad_last=False, receipt=None):
    """Execute a native SDE prefix with the released action-input constraint."""
    from instinct_compress.models.cosmos3_distill_rollout import validate_times

    times = validate_times(times)
    if isinstance(prefix_steps, bool) or prefix_steps not in (1, 2, 3, 4):
        raise ValueError("prefix_steps must be an integer from 1 through 4")
    levels = list(times[:prefix_steps]) + [0.0]
    sampler = ActionAwareFixedStepSampler(condition, t_list=levels)
    calls, clocks, graphs = 0, [], []

    def velocity(noise, timestep):
        nonlocal calls
        calls += 1
        clocks.append(float(timestep.item()))
        if calls > prefix_steps:
            raise RuntimeError("Action-aware native sampler exceeded its call budget")
        state = torch.stack([value.detach() for value in noise])
        with torch.set_grad_enabled(grad_last and calls == prefix_steps):
            time = timestep.reshape(-1).expand(len(state)) / 1000.
            prediction = native_action_velocity(adapter, state, time, condition)
            graphs.append(prediction.requires_grad)
            return list(prediction.unbind(0))

    device = condition.initial_noise.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.inference_mode(False), torch.random.fork_rng(devices=devices), torch.set_grad_enabled(grad_last):
        result = sampler(velocity, list(condition.initial_noise.detach().clone().unbind(0)),
            seed=list(condition.seeds), num_steps=prefix_steps,
            condition_reference=list(condition.reference.detach().unbind(0)),
            condition_mask=list(action_condition_mask(condition).unbind(0)))
        result = torch.stack(result)
    if calls != prefix_steps:
        raise RuntimeError("Action-aware native sampler did not execute its declared calls")
    if receipt is not None:
        receipt.update(callbacks=calls, network_branches=calls * (1 if adapter.guidance == 1 else 2),
            callback_timesteps=clocks, times=levels, guidance=float(adapter.guidance),
            gradient="last_callback" if grad_last else "none",
            sampler="native FixedStepSampler with zero action padding",
            native_padding_policy=False, action_padding_policy="droid_rf_training_zero_action_padding",
            callback_parameter_graphs=graphs, output_requires_grad=result.requires_grad,
            callback_padding_max=[max((_maximum(value) for value in values), default=0.)
                                  for values in sampler.callback_padding])
    return result


def _identity(value):
    item = value.detach().contiguous().cpu()
    return {"shape": list(item.shape), "dtype": str(item.dtype),
            "sha256": hashlib.sha256(item.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _prepared_identity(prepared):
    return {name: [_identity(value) for value in prepared[index]] for name, index in
            (("noise", 4), ("condition_reference", 5), ("condition_mask", 6))}


def _maximum(value):
    return float(value.float().abs().max().item()) if value.numel() else 0.0


@dataclass
class PaddingAudit:
    original_preparations: list = field(default_factory=list)
    effective_preparations: list = field(default_factory=list)
    layouts: list = field(default_factory=list)
    branches: list = field(default_factory=list)
    hooks_restored: bool = False

    def report(self):
        if len(self.original_preparations) != 1:
            raise ValueError("One completed native preparation is required per ablation receipt")
        original, effective = self.original_preparations[0], self.effective_preparations[0]
        layouts = self.layouts[0]
        untouched = all(torch.equal(layout.valid(a), layout.valid(b))
            for values_a, values_b in zip(original[4:7], effective[4:7], strict=True)
            for layout, a, b in zip(layouts, values_a, values_b, strict=True))
        return {
            "schema_version": 1,
            "policy": "droid_rf_training_zero_action_padding",
            "raw_action_dim": 8, "padded_action_dim": 64,
            "sampler_change": "padded_coordinates_are_zero_reference_constraints",
            "velocity_change": "project_only_padded_action_inputs_before_native_packing",
            "prepared_calls": len(self.original_preparations),
            "prepared_original": _prepared_identity(original),
            "prepared_effective": _prepared_identity(effective),
            "nonpadding_unchanged": untouched,
            "mask_delta_coordinates": [int((a != b).sum().item())
                for a, b in zip(original[6], effective[6], strict=True)],
            "initial_padding_max_before": [_maximum(layout.view(value)[:, 8:])
                for layout, value in zip(layouts, original[4], strict=True)],
            "initial_padding_max_after": [_maximum(layout.view(value)[:, 8:])
                for layout, value in zip(layouts, effective[4], strict=True)],
            "projected_velocity_branches": len(self.branches),
            "velocity_padding_max_before": [max((_maximum(v) for v in b[0]), default=0.0)
                for b in self.branches],
            "velocity_padding_max_after": [max((_maximum(v) for v in b[1]), default=0.0)
                for b in self.branches],
            "hooks_restored": self.hooks_restored,
        }


@contextmanager
def zero_action_padding(model):
    """Temporarily enforce zero input/transition padding for native DROID.

    Call under the native service lock, enclosing one raw generation request.
    No original preparation tensor or metadata is mutated. NativeCapture may
    be installed inside or outside this context; the audit retains both the
    original and effective preparation identities. The underlying sampler is
    not replaced. Its ordinary mask blend enforces zeros after every step.
    """
    audit = PaddingAudit()
    patches = []

    def preparing(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            prepared = original(*args, **kwargs)
            if len(prepared) != 7:
                raise ValueError("Unexpected native preparation contract")
            layouts = action_layouts(prepared[0], prepared[1])
            if any(len(values) != len(layouts) for values in prepared[4:7]):
                raise ValueError("Native prepared tensor count differs from metadata")
            for layout, mask in zip(layouts, prepared[6], strict=True):
                action_mask = layout.view(mask)
                if not torch.equal(action_mask, action_mask[:, :1].expand_as(action_mask)):
                    raise ValueError("Original native action conditioning must be frame-wise")
                if not torch.all((action_mask == 0) | (action_mask == 1)):
                    raise ValueError("Original native action conditioning must be binary")
            effective = tuple(prepared[:4]) + tuple(
                [layout.project(value, fill) for layout, value in zip(layouts, values, strict=True)]
                for values, fill in zip(prepared[4:7], (0, 0, 1), strict=True))
            audit.original_preparations.append(prepared)
            audit.effective_preparations.append(effective)
            audit.layouts.append(layouts)
            return effective
        return wrapped

    def velocity(original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if args or not {"noise_x", "sequence_plans", "gen_data_clean"} <= kwargs.keys():
                raise ValueError("Expected the keyword-only native velocity contract")
            layouts = action_layouts(kwargs["sequence_plans"], kwargs["gen_data_clean"])
            noise = kwargs["noise_x"]
            if len(noise) != len(layouts):
                raise ValueError("Native velocity sample count differs from metadata")
            projected = [layout.project(value) for layout, value in zip(layouts, noise, strict=True)]
            audit.branches.append((
                [layout.view(value)[:, 8:].detach().clone()
                 for layout, value in zip(layouts, noise, strict=True)],
                [layout.view(value)[:, 8:].detach().clone()
                 for layout, value in zip(layouts, projected, strict=True)]))
            return original(**{**kwargs, "noise_x": projected})
        return wrapped

    try:
        for name, factory in (("_prepare_inference_data", preparing), ("_get_velocity", velocity)):
            own, original = name in vars(model), getattr(model, name)
            patches.append((name, own, original))
            setattr(model, name, factory(original))
        yield audit
    finally:
        for name, own, original in reversed(patches):
            if own:
                setattr(model, name, original)
            else:
                delattr(model, name)
        audit.hooks_restored = True
