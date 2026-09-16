"""Opt-in pi0.5 execution-state recording and strict cross-process restoration.

Default Pi05CheckpointFrontend behavior is unchanged. A frozen state is bound
to one prompt, model, software/binary environment and registered token lengths.
"""
from __future__ import annotations

import importlib.metadata
import hashlib
from pathlib import Path

import numpy as np
import torch

from flash_rt.core import frozen_state as state
from flash_rt.frontends.torch.pi05_checkpoint import Pi05CheckpointFrontend
from flash_rt.frontends.torch.pi05_rtx import ENC_L


def observation_hashes(observations):
    result = []
    for observation in observations:
        item = {}
        for key in ("image", "wrist_image", "state"):
            value = np.ascontiguousarray(observation[key])
            item[key] = {"shape": list(value.shape), "dtype": value.dtype.str,
                         "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        result.append(state.digest(item))
    return result


class FrozenPi05Frontend(Pi05CheckpointFrontend):
    def __init__(self, checkpoint_dir, *, use_fp8=True, **kwargs):
        if not use_fp8 or kwargs.get("tokenizer_path") or kwargs.get("bf16_encoder_down_layers"):
            raise ValueError("frozen pi05 currently requires full FP8 and the checkpoint-default tokenizer")
        super().__init__(checkpoint_dir, use_fp8=True, **kwargs)
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("frozen pi05 requires SM120")
        if not hasattr(self.frontend.gemm, "export_algo_cache"):
            raise RuntimeError("rebuild flash_rt_kernels with frozen GEMM support")
        self.frontend.gemm.set_cache_policy("record")
        self._checkpoint_dir = Path(checkpoint_dir)
        self._state_prompt = None
        self._allowed_lengths = None
        self._seen_lengths = set()
        self._override_prepared = None
        self._state_calibration = None
        self._identity = None
        self._restored_parts = None

    def identity(self):
        if self._identity is None:
            import lerobot
            import transformers
            from huggingface_hub import try_to_load_from_cache
            from flash_rt import flash_rt_kernels, flash_rt_fa2
            root = Path(__file__).resolve().parents[2]
            def tree(folder):
                return state.digest({str(p.relative_to(folder)): state.file_hash(p)
                                     for p in sorted(folder.rglob("*.py"))})
            tokenizer = {}
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
                path = try_to_load_from_cache("google/paligemma-3b-pt-224", name)
                if not isinstance(path, str):
                    raise ValueError("missing pinned tokenizer file")
                tokenizer[name] = state.file_hash(path)
            self._identity = {
                "weights": state.file_hash(self._checkpoint_dir / "model.safetensors"),
                "processors": {p.name: state.file_hash(p) for p in sorted(self._checkpoint_dir.iterdir())
                               if p.is_file() and (p.name == "config.json" or p.name.startswith("policy_"))},
                "source_tree": tree(root),
                "processor_trees": {"lerobot": tree(Path(lerobot.__file__).parent),
                                    "transformers": tree(Path(transformers.__file__).parent)},
                "tokenizer": tokenizer,
                "packages": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "lerobot", "transformers", "tokenizers")},
                "binary": {"kernels": state.file_hash(flash_rt_kernels.__file__), "fa2": state.file_hash(flash_rt_fa2.__file__)},
                "gemm": self.frontend.gemm.algo_cache_identity(),
                "contract": {"computed": 50, "returned": 10, "nfe": 10, "views": self.frontend.num_views,
                             "layout": self.frontend._build_pipeline_weights()["fp8_layout"]}}
        return self._identity

    def set_prompt(self, prompt, state=None):
        if self._state_prompt is not None and prompt != self._state_prompt:
            raise ValueError("frozen execution state belongs to a different prompt")
        return super().set_prompt(prompt, state)

    def prepare(self, observation):
        if self._override_prepared is not None:
            return self._override_prepared, None
        return super().prepare(observation)

    def _set_tokens(self, ids):
        if self._allowed_lengths is not None and len(ids) not in self._allowed_lengths:
            raise ValueError(f"unregistered frozen token length: {len(ids)}")
        frontend = super()._set_tokens(ids)
        self._seen_lengths.add(len(ids))
        return frontend

    def calibrate(self, observations, *, percentile=99.9, max_samples=None, verbose=False):
        if self._restored_parts in ("both", "scales"):
            raise RuntimeError("restored scales are immutable; create a new state to recalibrate")
        values = [observations] if isinstance(observations, dict) else list(observations)
        if max_samples is not None:
            values = values[:max_samples]
        provenance = {"observations_sha256": observation_hashes(values), "percentile": percentile}
        if self._state_calibration is not None and provenance != self._state_calibration:
            raise ValueError("calibration data or percentile differs from frozen state")
        # First pass registers/tunes every calibration shape. Repeating with the
        # now-stable choices avoids measuring the initial heuristic algorithms.
        super().calibrate(values, percentile=percentile, verbose=verbose)
        super().calibrate(values, percentile=percentile, verbose=verbose)
        self._state_calibration = provenance

    calibrate_with_real_data = calibrate

    def register_profiles(self, observation, token_lengths):
        if self._allowed_lengths is not None or self._scales is None:
            raise RuntimeError("register profiles after calibration and before freezing")
        lengths = sorted(set(token_lengths))
        if any(type(n) is not int or not 1 <= n <= self.config.tokenizer_max_length for n in lengths):
            raise ValueError("invalid profile length")
        prepared, _ = super().prepare(observation)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for length in lengths:
                    self._override_prepared = {**prepared, "token_ids": np.resize(prepared["token_ids"], length)}
                    super().infer(observation)
        finally:
            self._override_prepared = None

    def save_state(self, path):
        if self._scales is None or self._state_calibration is None or not self._seen_lengths:
            raise RuntimeError("calibrate and register profiles before saving")
        gemm = self.frontend.gemm
        payload = {"schema": state.SCHEMA, "identity": self.identity(), "prompt": self._prompt,
            "calibration": self._state_calibration, "scale_bits": state.scale_bits(self._scales),
            "token_lengths": sorted(self._seen_lengths),
            "gemm": {"identity": gemm.algo_cache_identity(),
                     "entries": [[*row[:4], row[4].hex()] for row in gemm.export_algo_cache()]}}
        digest = state.save(path, payload)
        gemm.set_cache_policy("frozen")
        self._state_prompt = self._prompt
        self._allowed_lengths = set(self._seen_lengths)
        return digest

    def load_state(self, path, *, parts="both"):
        if parts not in ("both", "scales", "algorithms"):
            raise ValueError("parts must be both, scales or algorithms")
        if self._profiles or self._scales is not None or self.frontend.gemm.export_algo_cache():
            raise RuntimeError("load state into a fresh model before calibration or capture")
        payload = state.load(path, self.identity())
        gemm = self.frontend.gemm
        if payload["gemm"]["identity"] != gemm.algo_cache_identity():
            raise ValueError("GEMM environment differs")
        scales = state.decode_scales(payload["scale_bits"])
        # The final encoder layer ends immediately after producing decoder K/V.
        unused = {f"encoder_{name}_w_{ENC_L - 1}" for name in ("attn_o", "ffn_gate_up", "ffn_down")}
        expected = set(self.frontend._fp8_weights) - unused
        if set(scales) != expected:
            raise ValueError("frozen calibration layer set differs from the model")
        if parts != "scales":
            rows = [(*row[:4], bytes.fromhex(row[4])) for row in payload["gemm"]["entries"]]
            gemm.import_algo_cache(payload["gemm"]["identity"], rows)
        self.set_prompt(payload["prompt"])
        self._state_prompt = payload["prompt"]
        self._allowed_lengths = set(payload["token_lengths"])
        self._state_calibration = payload["calibration"]
        self._restored_parts = parts
        if parts != "algorithms":
            self._scales = scales
            self._calibration = {"samples": len(payload["calibration"]["observations_sha256"]),
                                 "percentile": payload["calibration"]["percentile"]}
        return payload
