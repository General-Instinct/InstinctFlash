"""VLA2 native BF16 vision and checkpoint-derived FP8 expert integration.

This staged component is not yet registered as a public Runtime backend.
"""
from contextlib import ExitStack
import json
from pathlib import Path


class NativeVla2Vision:
    """Execute the checkpoint's patched Qwen3 vision and all deepstack features."""

    def __init__(self, config, checkpoint, device):
        import torch
        from safetensors import safe_open
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        apply_lingbot_qwen3_vl_patch()
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        config._attn_implementation = 'eager'
        self.model = Qwen3VLVisionModel(config)
        root = Path(checkpoint)
        prefix = 'model.qwenvl_with_expert.qwenvl.model.visual.'
        index = root/'model.safetensors.index.json'
        with ExitStack() as stack:
            if index.exists():
                mapping = json.loads(index.read_text())['weight_map']
            else:
                sf = stack.enter_context(safe_open(str(root/'model.safetensors'), framework='pt', device='cpu'))
                mapping = {k:'model.safetensors' for k in sf.keys()}
            files, weights = {}, {}
            for key, filename in mapping.items():
                if not key.startswith(prefix):
                    continue
                if filename not in files:
                    files[filename] = stack.enter_context(safe_open(str(root/filename), framework='pt', device='cpu'))
                weights[key[len(prefix):]] = files[filename].get_tensor(key)
            self.model.load_state_dict(weights, strict=True)
        self.model.to(device=device, dtype=torch.bfloat16).eval()
        self.grid = torch.tensor([[1,16,16]]*3, device=device)
        self.layout = self.model.preprcess_grid_thw(self.grid)
        pos,rotary,cu,splits,maximum = self.layout
        if pos is None:
            pos = self.model.fast_pos_embed_interpolate(self.grid)
        self.capture_layout = (pos,rotary,cu.cpu(),splits,maximum)
        from .static_tensor_graph import StaticTensorGraph
        self.capture = StaticTensorGraph(lambda x: self._run(x,self.capture_layout),
                                         reference=lambda x: self._run(x,self.layout),
                                         name='vla2_bf16_vision')

    def __call__(self, patches):
        import torch
        merged, deepstack = self.capture(patches)
        return merged, deepstack

    def _run(self, patches, layout):
        import torch
        pos, rotary, cu, _, maximum = layout
        with torch.no_grad():
            merged, deepstack = self.model(
                patches.reshape(768,1536).to(device=self.grid.device, dtype=torch.bfloat16),
                grid_thw=self.grid, pos_embeds=pos, position_embeddings=rotary,
                cu_seqlens=cu, max_seqlen=maximum)
        features = [merged, *deepstack]
        if len(features) != 4 or any(tuple(v.shape) != (192,2560) for v in features):
            raise RuntimeError('VLA2 native vision must return merged and three deepstack features')
        if not torch.cuda.is_current_stream_capturing() and not all(bool(torch.isfinite(v).all()) for v in features):
            raise RuntimeError('VLA2 native vision returned nonfinite features')
        return merged, deepstack


class ExpertScaleAccumulator:
    """Retain each layer's maximum scale over every calibration denoise step."""

    def __init__(self, expert_scales, moe_scales, steps=10):
        import torch
        self.expert = expert_scales.reshape(-1,4)
        self.moe = moe_scales.reshape(-1)
        if len(self.expert) != len(self.moe):
            raise ValueError('expert and routed MoE layer counts differ')
        self.expert_max = torch.zeros_like(self.expert)
        self.moe_max = torch.zeros_like(self.moe)
        self.seen = [set() for _ in range(len(self.moe))]
        self.steps = steps

    def __call__(self, layer, step):
        import torch
        if step not in range(self.steps) or step in self.seen[layer]:
            raise ValueError('invalid or repeated calibration step')
        torch.maximum(self.expert_max[layer], self.expert[layer], out=self.expert_max[layer])
        torch.maximum(self.moe_max[layer], self.moe[layer], out=self.moe_max[layer])
        self.seen[layer].add(step)

    def commit(self):
        import torch
        if any(s != set(range(self.steps)) for s in self.seen):
            raise RuntimeError('incomplete expert calibration')
        for scales in (self.expert_max,self.moe_max):
            if not bool((torch.isfinite(scales) & (scales > 0)).all()):
                raise RuntimeError('invalid expert calibration scales')
        self.expert.copy_(self.expert_max)
        self.moe.copy_(self.moe_max)


class Vla2StagedEngine:
    """Live native vision plus FP16 prefill and real FP8 action/MoE kernels.

    Calibrate on the first observation for each token sequence. Scales aggregate
    all ten denoise steps; no historical calibration artifact is imported.
    """

    def __init__(self, frontend, moe, vision, use_cuda_graph=True):
        if frontend.use_cuda_graph:
            raise ValueError('construct frontend with use_cuda_graph=False; this route owns capture')
        if frontend.lm_prefill_precision != 'fp16':
            raise ValueError('native-vision V2 recipe requires the checkpoint FP16 prefill stack')
        self.frontend, self.moe, self.vision = frontend, moe, vision
        self.use_cuda_graph = use_cuda_graph
        self._tokens = None
        self._calibrated = False
        self._graph = None
        self.replays = 0
        self.graph_captures = 0
        self.prompt_graph_reuses = 0
        self.calibrations = 0
        frontend._routed_moe_fn = moe.routed_moe_fn
        frontend._ffn_xn_fp16 = True

    def set_prompt(self, token_ids):
        ids = tuple(int(i) for i in token_ids)
        if not ids:
            raise ValueError('VLA2 requires active language tokens')
        if ids == self._tokens:
            return
        self._calibrated = False
        try:
            self.frontend.set_prompt(list(ids))
            if not self.frontend._prompt_buffers_reused:
                self._graph = None
            self.moe.bind_router_fp16(self.frontend._s_xn.data_ptr())
            self.moe.bind_ffn_slots(self.frontend._exp_act_scales.data_ptr()+8,16)
        except Exception:
            # A partial rebuild or binding failure must not replay an old
            # graph against partially updated storage on a later call.
            self._graph = None
            self._tokens = None
            self.frontend.graph_captured = False
            raise
        self.frontend.graph_captured = self._graph is not None
        if self._graph is not None:
            self.prompt_graph_reuses += 1
        self._tokens = ids

    def infer_staged(self, patches, state, noise):
        import time
        import torch
        if self._tokens is None:
            raise RuntimeError('set_prompt must precede inference')
        fr = self.frontend
        t0 = time.perf_counter()
        # The low-level calibration kernels use stream zero. Keep every copy,
        # native vision call and scale observation ordered on that same stream.
        with torch.no_grad(), torch.cuda.stream(torch.cuda.default_stream(fr._x_t.device)):
            merged, deepstack = self.vision(patches)
            fr._vis_emb.copy_(merged)
            for target, source in zip(fr._ds_out,deepstack):
                target.copy_(source)
            fr.stage_inputs(patches,state,noise)
            if not self._calibrated:
                fr._run_lm_prefill_only(0)
                accumulator = ExpertScaleAccumulator(fr._exp_act_scales,self.moe.dn_act)
                self.moe.calibrate = True
                try:
                    fr._run_expert_only(0,calibrate=True,calibration_observer=accumulator)
                    accumulator.commit()
                finally:
                    self.moe.calibrate = False
                self._calibrated = True
                self.calibrations += 1
                fr._x_t.copy_(fr._noise_host)
                if self.use_cuda_graph and self._graph is None:
                    # Warmup and capture each mutate x_t; restore the caller's
                    # actual noise before the first serving replay below.
                    fr._run_lm_prefill_only(0)
                    fr._run_expert_only(0)
                    torch.cuda.synchronize()
                    stream = torch.cuda.Stream(device=fr._x_t.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(stream), torch.cuda.graph(graph,stream=stream):
                        fr._run_lm_prefill_only(stream.cuda_stream)
                        fr._run_expert_only(stream.cuda_stream)
                    torch.cuda.synchronize()
                    self._graph = graph
                    self.graph_captures += 1
                    fr.graph_captured = True
            fr._x_t.copy_(fr._noise_host)
            if self._graph is not None:
                self._graph.replay()
                self.replays += 1
            else:
                fr._run_lm_prefill_only(0)
                fr._run_expert_only(0)
            torch.cuda.synchronize()
            if not bool(torch.isfinite(fr._x_t).all()):
                raise RuntimeError('VLA2 FP8 action expert returned nonfinite actions')
            actions = fr._x_t.float().cpu().numpy().copy()
        elapsed = (time.perf_counter()-t0)*1000
        fr.latency_records.append(elapsed)
        return dict(actions=actions,latency_ms=elapsed)


class Vla2ActionGenerator:
    """Native sample_actions signature around the staged engine."""

    def __init__(self, engine, config):
        if (config.num_steps != 10 or config.n_action_steps != 50
                or config.max_action_dim != 55):
            raise ValueError('VLA2 FP8 recipe requires ten steps and 50x55 model actions')
        self.engine, self.config = engine, config

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, image_grid_thw=None):
        import torch
        if tuple(state.shape) != (1,55):
            raise ValueError('VLA2 FP8 requires one normalized 55-dimensional state')
        if img_masks.numel() != 3 or not bool(img_masks.bool().all()):
            raise ValueError('VLA2 FP8 requires three active cameras in native order')
        if images.numel() != 768*1536:
            raise ValueError('VLA2 FP8 requires native 3x256x1536 Qwen3 patches')
        if image_grid_thw is not None:
            grid = torch.as_tensor(image_grid_thw,device=state.device).reshape(-1,3)
            expected = torch.tensor([[1,16,16]]*3,device=state.device)
            if not torch.equal(grid,expected):
                raise ValueError('VLA2 FP8 requires three 1x16x16 image grids')
        if (lang_tokens.ndim != 2 or lang_tokens.shape[0] != 1
                or lang_masks.shape != lang_tokens.shape):
            raise ValueError('VLA2 token IDs and masks must match [1,length]')
        self.engine.set_prompt(lang_tokens[lang_masks.bool()].detach().cpu().tolist())
        if noise is None:
            noise = torch.randn((1,50,55),device=state.device,dtype=state.dtype)
        if tuple(noise.shape) != (1,50,55):
            raise ValueError('VLA2 FP8 noise must preserve the native 50x55 horizon')
        result = self.engine.infer_staged(images,state,noise)['actions']
        actions = torch.as_tensor(result,device=state.device)
        if tuple(actions.shape) != (50,55) or not bool(torch.isfinite(actions).all()):
            raise RuntimeError('VLA2 FP8 returned invalid normalized actions')
        return actions.unsqueeze(0)


def build_vla2_engine_loop(checkpoint, *, device=None, action_steps=10):
    """Reuse the V2 server's processors, robot normalization and episode reset."""
    import os
    import sys
    from types import SimpleNamespace
    import torch
    import yaml
    from lingbot_vla_v2_iwm.adapter import _source_root, _resolve_model_path, _LingBotVLAV2Loop
    if action_steps != 10:
        raise ValueError('VLA2 engine requires the declared ten-step schedule')
    dev = torch.device(device or 'cuda')
    if dev.type != 'cuda':
        raise ValueError('VLA2 engine requires CUDA')
    root = _source_root()
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    extra = dict(checkpoint.execution.extra or {})
    base = os.environ.get('QWEN3VL_PATH') or str(extra.get('tokenizer_repo') or 'Qwen/Qwen3-VL-4B-Instruct')
    os.environ['QWEN3VL_PATH'] = base
    model_path = _resolve_model_path(checkpoint)
    # Same offline tokenizer compatibility workaround as the native adapter.
    import transformers.tokenization_utils_base as tub
    if hasattr(tub.PreTrainedTokenizerBase, '_patch_mistral_regex'):
        tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(
            lambda cls, tokenizer, *args, **kwargs: tokenizer)
    from deploy.lingbot_vla_v2_policy import (
        LingbotVLAv2Server, PolicyPreprocessMixin, LingbotVLAV2Config,
        AutoConfig, build_processor,
    )
    from flash_rt.frontends.torch.vla2_thor import Vla2TorchFrontendThor
    from flash_rt.models.vla2.moe_engine import Vla2MoeEngine
    import flash_rt.flash_rt_kernels as fvk

    class EnginePolicy(PolicyPreprocessMixin, torch.nn.Module):
        def __init__(self, generator):
            super().__init__()
            # A plain generator keeps the native server's .to(BF16) from
            # rewriting FP8 engine weights; vision owns its BF16 module.
            self.model = generator
            self.feature_transform = None

    class EngineServer(LingbotVLAv2Server):
        def load_vla(self, path_to_pi_model):
            training = yaml.safe_load((Path(path_to_pi_model).parent.parent.parent/'lingbotvla_cli.yaml').read_text())
            kwargs = {**training['model'], **training['train']}
            config = LingbotVLAV2Config(**kwargs)
            for key,value in kwargs.items():
                if not hasattr(config,key):setattr(config,key,value)
            config.attention_implementation = 'eager'
            config.tokenizer_path = base
            self.config, self.model_name = config, 'qwen3vl'
            self.merge_qwen_config(AutoConfig.from_pretrained(base))
            if training['model'].get('vocab_size',0):
                config.vocab_size = training['model']['vocab_size']
            config.use_cache = True
            if (config.max_state_dim != 55 or config.max_action_dim != 55
                    or config.chunk_size != 50 or config.n_action_steps != 50
                    or config.num_steps != action_steps):
                raise ValueError('VLA2 FP8 requires state/action=55, chunk=50, ten denoise steps')
            self.processor = build_processor(base)
            self.language_tokenizer = self.processor.tokenizer
            self.data_config = SimpleNamespace(**training['data'])
            if self.robot_norm_path is None:
                self.robot_norm_path = self.data_config.norm_stats_file
            vision = NativeVla2Vision(config.vision_config,path_to_pi_model,dev)
            moe = Vla2MoeEngine.from_checkpoint(fvk,path_to_pi_model,mode='batched',
                                               schedule='batched',router_source='fp16')
            frontend = Vla2TorchFrontendThor(path_to_pi_model,use_cuda_graph=False,
                                             routed_moe_fn=moe.routed_moe_fn)
            frontend.load_lm_fp16_stack(path_to_pi_model)
            frontend.lm_prefill_precision = 'fp16'
            generator = Vla2ActionGenerator(Vla2StagedEngine(frontend,moe,vision),config)
            self.sample_actions_fn = generator.sample_actions
            self.vla = EnginePolicy(generator)
            return self.vla

    with torch.cuda.device(dev):
        server = EngineServer(str(model_path),use_length=50,chunk_ret=True,
                              use_bf16=True,use_fp32=False,use_compile=False)

    class EngineLoop(_LingBotVLAV2Loop):
        @property
        def graph_stats(self):
            engine = self._server.vla.model.engine
            return dict(captured=engine._graph is not None,replays=engine.replays,
                        vision_graph=engine.vision.capture.graph is not None,vision_self_check=engine.vision.capture.verdict,language_action_graph=engine._graph is not None,
                        language_action_captures=engine.graph_captures,
                        language_action_prompt_reuses=engine.prompt_graph_reuses,
                        expert_calibrations=engine.calibrations,
                        prompt_updates=engine.frontend.prompt_updates,
                        prompt_buffer_reuses=engine.frontend.prompt_buffer_reuses)

        def close(self):
            if self._server is not None:
                self._server.vla.model.engine.vision.capture.close()
            return super().close()

        def predict(self, observation, *, executed_action=None):
            with torch.cuda.device(dev):
                return super().predict(observation)

    return EngineLoop(server,root,robot=str(extra.get('robot') or 'robotwin'))
