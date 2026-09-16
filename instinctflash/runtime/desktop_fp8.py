"""Architecture identities for explicit desktop FP8 recipes; no Torch import."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionTarget:
    capability: tuple[int, int]
    prefix: str
    label: str
    device_name: str

    @property
    def executor(self):
        return f"{self.prefix}_torch_fp8"

    @property
    def receipt_attribute(self):
        return f"_{self.prefix}_fp8_recipe"

    @property
    def builder(self):
        return f"build_{self.prefix}_fp8"

    @property
    def linear_class(self):
        return f"{self.label}FP8Linear"


SM89 = ProjectionTarget((8, 9), "sm89", "SM89", "NVIDIA GeForce RTX 4090")
SM120 = ProjectionTarget((12, 0), "sm120", "SM120", "NVIDIA GeForce RTX 5090")
TARGETS = (SM89, SM120)
EXECUTORS = frozenset(target.executor for target in TARGETS)


def target_for_capability(capability):
    return next((target for target in TARGETS if tuple(capability) == target.capability), None)


def backend_for_capability(capability):
    target = target_for_capability(capability)
    if target is SM89:
        from . import sm89_fp8
        return sm89_fp8
    if target is SM120:
        from . import sm120_fp8
        return sm120_fp8
    raise ValueError(f"No desktop FP8 recipe for CUDA capability {capability}")
