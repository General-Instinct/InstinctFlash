"""YAML-configurable Layer 2-6 optimization pipelines."""

from instinctwm.config.compiler import resolve_pipeline
from instinctwm.config.loader import PRESETS, load_config, preset_path
from instinctwm.config.registry import PassRegistry, default_registry
from instinctwm.config.schema import (
    ConfigurationError,
    InstallPhase,
    OptimizationLayer,
    ParameterSpec,
    PassDefinition,
    PassMaturity,
    PassMode,
    PipelineConfig,
    PipelinePolicy,
    ResolvedPipeline,
)

__all__ = [
    "ConfigurationError", "InstallPhase", "OptimizationLayer", "ParameterSpec",
    "PassDefinition", "PassMaturity", "PassMode", "PassRegistry", "PipelineConfig",
    "PipelinePolicy", "ResolvedPipeline", "default_registry", "load_config", "preset_path",
    "resolve_pipeline", "available_presets",
]


def available_presets() -> tuple[str, ...]:
    return tuple(sorted(PRESETS))
