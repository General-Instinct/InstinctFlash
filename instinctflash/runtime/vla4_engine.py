"""Thor VLA4 engine with the upstream server's robot processing contract.

The upstream server owns resize, FeatureTransform, normalization, reset and
controller slicing. Only its normalized action generator is replaced. Importing
this module does not import the model stack or initialize CUDA.
"""
from pathlib import Path
import os
import sys
from types import SimpleNamespace


class Vla4ActionGenerator:
    """The sample_actions boundary consumed by upstream PolicyPreprocessMixin."""

    def __init__(self, frontend, vision=None):
        self.frontend = frontend
        self.vision = vision
        self._token_ids = None

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       vlm_causal=False, noise=None, num_steps=None):
        import torch
        if num_steps != 10 or vlm_causal:
            raise ValueError('VLA4 Thor engine requires ten steps and noncausal policy attention')
        if tuple(state.shape) != (1, 75):
            raise ValueError('VLA4 engine requires one normalized 75-dimensional state')
        if img_masks.numel() != 3 or not bool(img_masks.bool().all()):
            raise ValueError('VLA4 engine requires three active cameras in native processor order')
        if images.numel() != 3 * 256 * 1176:
            raise ValueError('VLA4 engine requires native Qwen patches from three 224x224 views')
        if (lang_tokens.ndim != 2 or lang_tokens.shape[0] != 1
                or lang_tokens.shape != lang_masks.shape):
            raise ValueError('VLA4 tokens and masks must have matching [1, length] shapes')
        ids = tuple(lang_tokens[lang_masks.bool()].detach().cpu().tolist())
        if not ids:
            raise ValueError('VLA4 engine requires active language tokens')
        if ids != self._token_ids:
            self.frontend.set_prompt(list(ids))
            self._token_ids = ids
        if noise is None:
            noise = torch.randn((1, 50, 75), device=state.device, dtype=state.dtype)
        if tuple(noise.shape) != (1, 50, 75):
            raise ValueError('VLA4 engine noise must preserve the native 50x75 horizon')
        kwargs = {} if self.vision is None else {'vision_embeddings': self.vision(images)}
        result = self.frontend.infer_staged(images, state, noise, **kwargs)['actions']
        actions = torch.as_tensor(result, device=state.device)
        if tuple(actions.shape) != (50, 75) or not bool(torch.isfinite(actions).all()):
            raise RuntimeError('VLA4 engine returned invalid normalized actions')
        return actions.unsqueeze(0)


class NativeVla4Vision:
    """Keep the native BF16 vision tower's range; FP16 overflows on real inputs."""

    def __init__(self, config, checkpoint, device):
        import torch
        from safetensors import safe_open
        from lingbotvla.models.vla.pi0.qwenvl_in_vla import Qwen2_5_VisionTransformerPretrainedModel
        # The qualified Thor native stack uses Qwen's eager vision attention;
        # retain it without depending on a separately installed flash_attn wheel.
        config._attn_implementation = 'eager'
        self.model = Qwen2_5_VisionTransformerPretrainedModel(config)
        prefix = 'model.qwenvl_with_expert.qwenvl.visual.'
        with safe_open(str(Path(checkpoint)/'model.safetensors'), framework='pt', device='cpu') as sf:
            weights = {key[len(prefix):]:sf.get_tensor(key) for key in sf.keys() if key.startswith(prefix)}
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device=device, dtype=torch.bfloat16).eval()
        self.grid = torch.tensor([[1,16,16]]*3, device=device)
        self.layout = self.model.preprcess_grid_thw(self.grid)
        self.capture_layout = tuple(v.to(device=device) if i == 1 else
                                    tuple(v.detach().cpu().tolist()) if i in (2,3) else v
                                    for i,v in enumerate(self.layout))
        from .static_tensor_graph import StaticTensorGraph
        self.capture = StaticTensorGraph(lambda x: self._run(x,self.capture_layout),
                                         reference=lambda x: self._run(x,self.layout),
                                         name='vla4_bf16_vision')

    def __call__(self, patches):
        import torch
        return self.capture(patches)

    def _run(self, patches, layout):
        import torch
        rotary, window, cu_window, cu = layout
        with torch.no_grad():
            return self.model(patches.reshape(-1,1176).to(dtype=torch.bfloat16),
                grid_thw=self.grid, rotary_pos_emb=rotary, window_index=window,
                cu_window_seqlens=cu_window, cu_seqlens=cu)


def build_vla4_engine_loop(checkpoint, *, device=None, action_steps=10):
    """Build native processors and an FP8 generator without a second full model."""
    import torch
    import yaml
    from lingbot_vla_iwm.adapter import (
        _source_root, _resolve_model_path, _resolve_norm_stats, _LingBotVLA4BLoop,
    )
    if action_steps != 10:
        raise ValueError('VLA4 Thor engine requires the declared ten-step schedule')
    dev = torch.device(device or 'cuda')
    if dev.type != 'cuda':
        raise ValueError('VLA4 Thor engine requires a CUDA device')
    root = _source_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    extra = dict(checkpoint.execution.extra or {})
    os.environ['QWEN25_PATH'] = os.environ.get('QWEN25_PATH') or str(
        extra.get('tokenizer_repo') or 'Qwen/Qwen2.5-VL-3B-Instruct')
    model_path = _resolve_model_path(checkpoint)
    norm_path = _resolve_norm_stats(checkpoint, root)

    from deploy.lingbot_vla_policy import (
        LingbotVLAServer, PolicyPreprocessMixin, PreTrainedConfig,
        AutoConfig, merge_qwen_config, build_processor,
    )
    from flash_rt.frontends.torch.vla4b_thor import Vla4bTorchFrontendThor

    class EnginePolicy(PolicyPreprocessMixin, torch.nn.Module):
        def __init__(self, generator):
            super().__init__()
            self.model = generator
            self.feature_transform = None

    class EngineServer(LingbotVLAServer):
        def load_vla(self, path_to_pi_model):
            # Mirror the native load_vla's configuration/processor construction;
            # avoid instantiating/loading LingBotVlaInferencePolicy weights.
            config = PreTrainedConfig.from_pretrained(path_to_pi_model)
            training = yaml.safe_load((Path(path_to_pi_model)/'lingbotvla_cli.yaml').read_text())
            kwargs = {**training['model'], **training['train']}
            config.__dict__.update({k:v for k,v in kwargs.items() if not hasattr(config,k)})
            config.attention_implementation = 'eager'
            base = os.environ['QWEN25_PATH']
            config.tokenizer_path = base
            config = merge_qwen_config(config, AutoConfig.from_pretrained(base))
            if training['model'].get('vocab_size', 0):
                config.vocab_size = training['model']['vocab_size']
            config.use_cache = True
            if (config.max_state_dim != 75 or config.max_action_dim != 75
                    or config.chunk_size != 50 or config.n_action_steps != 50):
                raise ValueError('VLA4 engine requires checkpoint geometry state=75, action=75, chunk=50')
            self.processor = build_processor(base)
            self.language_tokenizer = self.processor.tokenizer
            data_config = SimpleNamespace(**training['data'])
            for name in ('max_state_dim', 'max_action_dim', 'resize_imgs_with_padding',
                         'tokenizer_max_length'):
                setattr(data_config, name, getattr(config, name))
            self.data_config, self.config = data_config, config
            frontend = Vla4bTorchFrontendThor(path_to_pi_model)
            vision = NativeVla4Vision(config.vision_config, path_to_pi_model, dev)
            return EnginePolicy(Vla4ActionGenerator(frontend, vision))

    with torch.cuda.device(dev):
        server = EngineServer(str(model_path),
            use_length=int(extra.get('use_length') or 25), robot_norm_path=str(norm_path),
            num_denoising_step=action_steps, use_compile=False)
    # Reuse the native byte-checked image path independently of FP8 arithmetic.
    # A matched FP8 pair preserves action bytes and reduces Thor latency by ~4%.
    gpu_preprocess = None
    if os.environ.get("IFL_VLA4B_GPU_PREPROCESS", "1").lower() in {"1", "true", "yes", "on"}:
        from lingbot_vla_iwm.image_preprocess import install_gpu_image_preprocess
        gpu_preprocess = install_gpu_image_preprocess(server, device=dev)

    class EngineLoop(_LingBotVLA4BLoop):
        @property
        def graph_stats(self):
            frontend = self._server.vla.model.frontend
            return dict(super().graph_stats, captured=frontend.graph_captured,
                        replays=len(frontend.latency_records) if frontend.graph_captured else 0,
                        vision_graph=self._server.vla.model.vision.capture.graph is not None,
                        vision_self_check=self._server.vla.model.vision.capture.verdict,
                        language_action_graph=frontend.graph_captured,
                        prompt_updates=getattr(frontend, "prompt_updates", None),
                        prompt_buffer_reuses=getattr(frontend, "prompt_buffer_reuses", None),
                        prompt_graph_captures=getattr(frontend, "prompt_graph_captures", None))

        def close(self):
            if self._server is not None:
                self._server.vla.model.vision.capture.close()
            return super().close()

        def predict(self, observation, *, executed_action=None):
            # This family is stateless between chunks, like its native loop.
            with torch.cuda.device(dev):
                return super().predict(observation)

    return EngineLoop(server, root, robot=str(extra.get('robot') or 'robotwin'),
                      gpu_preprocess=gpu_preprocess)
