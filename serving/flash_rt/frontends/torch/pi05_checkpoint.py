"""SM120 execution with the checkpoint's actual LeRobot processor contract.

State-containing token IDs, image preprocessing, action normalization and the
computed chunk length are owned by the checkpoint. Only the executed/returned
action horizon is the qualified LIBERO operating-point choice (10).
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx


class Pi05CheckpointFrontend:
    requires_state = True

    def __init__(self, checkpoint_dir, num_views=2, use_fp8=True, hardware="rtx_sm120",
                 action_horizon=10, bf16_encoder_down_layers=(), tokenizer_path=None):
        if type(action_horizon) is not int or action_horizon != 10:
            raise ValueError("pi05 checkpoint path qualifies action_horizon=10 only")
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        checkpoint_dir = Path(checkpoint_dir)
        self.config = PreTrainedConfig.from_pretrained(checkpoint_dir)
        if not isinstance(self.config, PI05Config):
            raise ValueError("not a pi05 checkpoint")
        self.config.compile_model = False
        self.config.device = "cuda"
        self.config.n_action_steps = action_horizon
        self.config.validate_features()
        if self.config.chunk_size != 50 or self.config.num_inference_steps != 10:
            raise ValueError("qualification requires checkpoint chunk_size=50 and NFE=10")
        if (tuple(self.config.input_features["observation.state"].shape) != (8,) or
                tuple(self.config.output_features["action"].shape) != (7,)):
            raise ValueError("this SM120 checkpoint frontend requires LIBERO state dim 8 and action dim 7")
        processor = json.loads((checkpoint_dir / "policy_preprocessor.json").read_text())
        if not any(s["registry_name"] == "pi05_prepare_state_tokenizer_processor_step" for s in processor["steps"]):
            raise ValueError("missing checkpoint state-token processor")
        overrides = {"device_processor": {"device": "cuda"}}
        if tokenizer_path:
            overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
        self._pre, self._post = make_pre_post_processors(self.config,
            pretrained_path=str(checkpoint_dir), preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cuda"}})
        anchor = torch.empty(0, device="cuda")
        policy = SimpleNamespace(config=self.config, parameters=lambda: iter([anchor]))
        self._preprocess_images = lambda batch: PI05Policy._preprocess_images(policy, batch)
        self.frontend = Pi05TorchFrontendRtx(checkpoint_dir, num_views=num_views,
            chunk_size=self.config.chunk_size, max_prompt_len=self.config.tokenizer_max_length,
            use_fp8=use_fp8, hardware=hardware, bf16_encoder_down_layers=bf16_encoder_down_layers,
            action_output="normalized", inputs_normalized=True)
        self.action_horizon = action_horizon
        self._prompt = None
        self._profiles = OrderedDict()
        self._scales = None
        self._calibration = None

    def set_prompt(self, prompt, state=None):
        if self.frontend is None:
            raise RuntimeError("pi05 frontend is closed")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("pi05 requires a task prompt")
        if prompt != self._prompt:
            self._release_profiles()
            self.frontend.pipeline = None
            self.frontend.calibrated = False
            self._scales = None
        self._prompt = prompt

    def prepare(self, observation):
        if self._prompt is None:
            raise RuntimeError("set_prompt must be called first")
        if "state" not in observation:
            raise ValueError("this pi05 checkpoint requires an 8-dimensional robot state")
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("pi05 state must be a finite vector of shape (8,)")
        batch = {"observation.state": torch.from_numpy(state), "task": self._prompt}
        for target, source in (("image", "image"), ("image2", "wrist_image")):
            value = observation[source] if source in observation else observation["images"][0 if source == "image" else 1]
            image = np.asarray(value)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("checkpoint processor expects RGB uint8 images")
            batch[f"observation.images.{target}"] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        batch = self._pre(batch)
        images, masks = self._preprocess_images(batch)
        active = [image[0].permute(1, 2, 0).float().cpu().numpy()
                  for image, mask in zip(images, masks) if bool(mask.item())]
        if len(active) != self.frontend.num_views:
            raise ValueError("checkpoint active camera count differs from SM120 frontend")
        ids = batch["observation.language.tokens"][batch["observation.language.attention_mask"].bool()].cpu().numpy()
        return {"images": active, "token_ids": ids}, batch

    def _set_tokens(self, ids):
        fe = self.frontend
        size = len(ids)
        if size in self._profiles:
            pipeline, stream = self._profiles.pop(size)
            self._profiles[size] = (pipeline, stream)
            fe.pipeline, fe._graph_torch_stream = pipeline, stream
            fe.current_prompt_len = size
            fe.calibrated = fe.graph_recorded = True
        else:
            fe.pipeline = None
            fe.calibrated = fe.graph_recorded = False
        fe.set_prompt(ids)
        return fe

    def _remember(self):
        fe = self.frontend
        self._profiles[fe.current_prompt_len] = (fe.pipeline, fe._graph_torch_stream)
        while len(self._profiles) > 8:
            _, (pipeline, stream) = self._profiles.popitem(last=False)
            stream.synchronize()
            pipeline._graph.close()

    def _release_profiles(self):
        for pipeline, stream in self._profiles.values():
            stream.synchronize()
            pipeline._graph.close()
        self._profiles.clear()

    def calibrate(self, observations, *, percentile=99.9, max_samples=None, verbose=False):
        from flash_rt.core.calibration import accumulate_amax
        observations = [observations] if isinstance(observations, dict) else list(observations)
        if max_samples is not None:
            observations = observations[:max_samples]
        if not observations or not 0 <= percentile <= 100:
            raise ValueError("calibration requires real samples and a percentile in [0, 100]")
        vectors, names = [], None
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(5090120)
            for obs in observations:
                prepared, _ = self.prepare(obs)
                fe = self._set_tokens(prepared["token_ids"])
                fe.calibrated = False
                fe.pipeline.fp8_calibrated = False
                fe._zero_pipeline_scales()
                fe.calibrate(prepared)
                current = sorted(fe.pipeline.fp8_act_scales)
                if names is None:
                    names = current
                if current != names:
                    raise ValueError("calibration layer set changed between samples")
                vectors.append(np.array([fe.pipeline.fp8_act_scales[k].download_new((1,), np.float32)[0]
                                         for k in names], dtype=np.float32))
                self._remember()
        values = accumulate_amax(vectors, percentile)
        self._scales = dict(zip(names, values))
        self._calibration = {"samples": len(observations), "percentile": percentile}
        for pipeline, _ in self._profiles.values():
            self._install_scales(pipeline)

    calibrate_with_real_data = calibrate

    def _install_scales(self, pipeline):
        if self._scales is None:
            raise RuntimeError("fixed real-data calibration is required before inference")
        for name, value in self._scales.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid calibrated scale: {name}")
            pipeline._fp8_scale_buf(name).upload(np.array([value], dtype=np.float32))
        pipeline.fp8_calibrated = True

    def infer(self, observation):
        prepared, _ = self.prepare(observation)
        fe = self._set_tokens(prepared["token_ids"])
        self._install_scales(fe.pipeline)
        if not fe.calibrated:
            # New state-token lengths need new graphs, never new calibration data.
            # Graph warmup must not advance the episode's diffusion-noise RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                fe._graph_torch_stream = torch.cuda.Stream()
                with torch.cuda.stream(fe._graph_torch_stream):
                    stream = fe._graph_torch_stream.cuda_stream
                    fe._copy_tensor_to_pipeline_buf_stream(fe._stack_images(prepared), fe.pipeline.input_images_buf, stream)
                    noise = torch.zeros_like(fe._noise_buf)
                    fe._copy_tensor_to_pipeline_buf_stream(noise, fe.pipeline.input_noise_buf, stream)
                    fe.pipeline.record_infer_graph(external_stream_int=stream)
                fe.calibrated = fe.graph_recorded = True
                self._remember()
        normalized = fe.infer(prepared)["actions"]
        actions = self._post(torch.as_tensor(normalized[:, :7], device="cuda"))
        return {"actions": actions[:self.action_horizon].float().cpu().numpy()}

    @property
    def _noise_buf(self):
        return self.frontend._noise_buf

    @property
    def calibrated(self):
        return self._scales is not None

    @property
    def pipeline(self):
        return self.frontend.pipeline

    def close(self):
        self._release_profiles()
        if self.frontend is not None:
            self.frontend.pipeline = None
        self.frontend = None
        self._scales = self._prompt = None
        self._pre = self._post = self._preprocess_images = None
