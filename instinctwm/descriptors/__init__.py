"""Descriptors — the declared facts the rest of the stack is allowed to read.

Two kinds, and the split is load-bearing:

    AdapterSpec / ExecutionDescriptor / StateDescriptor   CHECKPOINT-SCOPED. Immutable, identical
        on every box the checkpoint runs on. "This model has two streams; the video stream is
        positive-only; KV is committed once per cycle."

    DeploymentSpec                                        SITE-SCOPED. The same checkpoint is one
        GPU here and eight there, and one caller wants pixels while the next wants only actions.

A pass that read a deployment fact off `AdapterSpec` would be asking the model author to declare
something they cannot know. See `deployment.py` for why that boundary earned its own type.
"""
from instinctwm.descriptors.deployment import DeploymentSpec
from instinctwm.descriptors.model import *  # noqa: F401,F403

__all__ = ["DeploymentSpec"]
