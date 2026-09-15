"""Native fixed-step RF trajectories for DROID distribution distillation.

NVIDIA's actual FixedStepSampler owns the latent arithmetic, fresh-noise seed
convention and conditioning blend. Training differentiates one selected prefix
exit; evaluation executes all four calls through the same implementation.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch


def validate_times(times):
    times = tuple(float(value) for value in times)
    if (len(times) != 5 or times[0] != 1.0 or times[-1] != 0.0
            or not all(math.isfinite(t) for t in times)
            or not all(a > b for a, b in zip(times, times[1:]))):
        raise ValueError("DROID distillation requires four decreasing calls from 1 to 0")
    return times


def native_condition_mask(condition, raw_dim=8):
    """Recover the native frame mask, distinct from the adapter's loss mask.

    The original serving sampler re-noises padded action coordinates at its
    intermediate transitions. The adapter's generic training mask additionally
    preserves those coordinates; using that mask would change native Fixed4.
    """
    mask = condition.preserve_mask.detach().clone()
    action = mask[:, condition.action_offset:].reshape(len(mask), *condition.action_shape)
    if not torch.equal(action[..., :raw_dim], action[..., :1].expand_as(action[..., :raw_dim])):
        raise ValueError("native DROID action conditioning must be frame-wise")
    action[..., raw_dim:] = action[..., :1]
    return mask


def resample_native_condition(adapter, condition, seed):
    """Regenerate native initial noise without repeating VAE/text preparation.

    Each vision item uses FP32 arch-invariant noise, action noise uses native
    model dtype, and each modality restarts the same request seed. Fixed input
    coordinates retain the actual prepared native initial values and rounding.
    """
    if len(condition.seeds) != 1 or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("one prepared DROID observation and a uint32 seed required")
    from cosmos_framework.utils import misc

    model = adapter.model
    parts = []
    for value in condition.gen_data_clean.x0_tokens_vision:
        parts.append(misc.arch_invariant_rand(tuple(value.shape),
            model.tensor_kwargs_fp32["dtype"], model.tensor_kwargs_fp32["device"], seed).flatten())
    action = misc.arch_invariant_rand(condition.action_shape,
        model.tensor_kwargs["dtype"], model.tensor_kwargs["device"], seed)
    action[:, 8:] = 0
    parts.append(action.flatten())
    initial = torch.cat(parts).reshape_as(condition.initial_noise)
    initial = torch.where(condition.preserve_mask.bool(), condition.initial_noise, initial)
    return replace(condition, initial_noise=initial.detach(), seeds=(seed,))


def native_velocity(adapter, latent, time, condition, *, guidance=None):
    """Use native packing/masks/CFG; do not pre-round or erase padded inputs."""
    if latent.shape != condition.reference.shape or time.shape != (len(latent),):
        raise ValueError("native velocity shape/time mismatch")
    scale = float(adapter.guidance if guidance is None else guidance)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("invalid guidance")
    timestep = time.reshape(-1, 1).float() * float(
        adapter.model.config.rectified_flow_inference_config.num_train_timesteps)
    with adapter._bound_heads():
        def run(tokens):
            return adapter.model._get_velocity(
                noise_x=list(latent.unbind(0)), timestep=timestep, text_tokens=tokens,
                sequence_plans=condition.sequence_plans, gen_data_clean=condition.gen_data_clean,
                skip_text_tokens=False)
        conditional = run(condition.cond_tokens)
        if scale == 1.0:
            return torch.stack(conditional)
        unconditional = run(condition.uncond_tokens)
        return torch.stack([u + scale * (c - u)
                            for c, u in zip(conditional, unconditional, strict=True)])


def native_clean(adapter, latent, time, condition, *, guidance=None):
    velocity = native_velocity(adapter, latent, time, condition, guidance=guidance)
    # Score conversion uses the FP32 sigma tensor, as in DMD2RF._velocity_to_x0.
    # The student generation loop separately retains native FixedStepSampler's
    # scalar/BF16 arithmetic, which is part of its serving contract.
    clean = latent.float() - time.reshape(-1, 1).float() * velocity.float()
    mask = native_condition_mask(condition).to(clean)
    return mask * condition.reference.to(clean) + (1 - mask) * clean


def rollout_native_fixed(adapter, condition, *, times=(1., .75, .5, .25, 0.),
                         prefix_steps=4, grad_last=False, receipt=None):
    """Execute the actual serving sampler with a chosen training prefix exit."""
    times = validate_times(times)
    if isinstance(prefix_steps, bool) or prefix_steps not in (1, 2, 3, 4):
        raise ValueError("prefix_steps must be an integer from 1 through 4")
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler

    levels = list(times[:prefix_steps]) + [0.0]
    sampler = FixedStepSampler(levels, sample_type="sde", num_train_timesteps=1000.)
    calls, clocks, graphs = 0, [], []

    def velocity(noise, timestep):
        nonlocal calls
        calls += 1
        clocks.append(float(timestep.item()))
        if calls > prefix_steps:
            raise RuntimeError("native fixed sampler exceeded the declared call budget")
        state = torch.stack([value.detach() for value in noise])
        with torch.set_grad_enabled(grad_last and calls == prefix_steps):
            time = timestep.reshape(-1).expand(len(state)) / 1000.
            prediction = native_velocity(adapter, state, time, condition)
            graphs.append(prediction.requires_grad)
            return list(prediction.unbind(0))

    device = condition.initial_noise.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Native sampling uses manual_seed(seed + step); isolate this side effect
    # from optimizer/data/score-noise RNG and restore it even on exceptions.
    with torch.inference_mode(False), torch.random.fork_rng(devices=devices), torch.set_grad_enabled(grad_last):
        result = sampler(velocity, list(condition.initial_noise.detach().clone().unbind(0)),
            seed=list(condition.seeds), num_steps=prefix_steps,
            condition_reference=list(condition.reference.detach().unbind(0)),
            condition_mask=list(native_condition_mask(condition).unbind(0)))
        result = torch.stack(result)
    if calls != prefix_steps:
        raise RuntimeError("native fixed sampler did not execute every declared call")
    if receipt is not None:
        receipt.update(callbacks=calls, network_branches=calls * (1 if adapter.guidance == 1 else 2),
                       callback_timesteps=clocks, times=levels, guidance=float(adapter.guidance),
                       gradient="last_callback" if grad_last else "none",
                       sampler="native FixedStepSampler", native_padding_policy=True,
                       callback_parameter_graphs=graphs, output_requires_grad=result.requires_grad)
    return result


def balanced_action_mask(batch, raw_dim=8):
    """Equal joint/gripper and first/all horizon weights, in physical units.

    This is an explicitly action-focused objective. It does not average the
    much larger generated-video tensor into an action quality loss.
    """
    mask = torch.zeros_like(batch.loss_mask)
    action = mask[:, batch.condition.action_offset:].reshape(len(mask), *batch.condition.action_shape)
    valid = batch.loss_mask[:, batch.condition.action_offset:].reshape_as(action)
    rows = valid[..., :raw_dim].sum(-1) > 0
    for b in range(len(mask)):
        indices = rows[b].nonzero(as_tuple=True)[0]
        if len(indices) != 32:
            raise ValueError("expected exactly 32 generated DROID action rows")
        action[b, indices, :7] = .25 / (32 * 7)
        action[b, indices, 7] = .25 / 32
        action[b, indices[0], :7] += .25 / 7
        action[b, indices[0], 7] += .25
    return mask


def renoise_generated(generated, condition, time, noise):
    """Noised generated examples for teacher/fake queries, with fixed context."""
    if generated.shape != noise.shape or time.shape != (len(generated),):
        raise ValueError("invalid score-noising shapes")
    t = time.reshape(-1, 1).float()
    noised = (1 - t) * generated.detach().float() + t * noise.float()
    mask = native_condition_mask(condition).bool()
    return torch.where(mask, condition.reference.to(noised), noised)
