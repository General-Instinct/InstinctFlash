"""Shared planning/runtime memory policy for the LingBot-VA serving stack.

This module deliberately does not import Torch: a plan can describe residency
before importing the model's environment. Eligibility is not a device-fit result.
"""


def needs_prompt_encoder_staging(capability, total_memory):
    """Retain reset-only T5 residency on memory-constrained Ada/Blackwell GPUs."""
    return tuple(capability) in {(8, 9), (12, 0)} and 0 < total_memory <= 40 << 30
