"""GR00T native processing with FP8 text/VLSA and native-weight BF16 DiT."""


class GrootActionGenerator:
    """Replace only action generation; the native camera/LLM runs every call."""

    def __init__(self, head, frontend, runner):
        config=head.config
        if (head.num_inference_timesteps!=4 or head.num_timestep_buckets!=1000
                or config.action_horizon!=40 or config.max_action_dim!=132
                or config.state_history_length!=1 or not config.add_pos_embed
                or not config.use_alternate_vl_dit):
            raise ValueError('GR00T engine requires native 40x132 actions, one state frame and four steps')
        self.head,self.frontend,self.runner=head,frontend,runner

    def get_action(self, backbone_output, action_input, options=None):
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        state=action_input.state
        if tuple(state.shape)!=(1,1,132):
            raise ValueError('GR00T engine requires one normalized 132-dimensional state')
        embodiment=action_input.embodiment_id
        if embodiment.numel()!=1 or int(embodiment.item())!=self.frontend._embodiment_id:
            raise ValueError('GR00T observation embodiment differs from loaded engine weights')
        with torch.no_grad():
            features=self.runner(backbone_output.backbone_features,backbone_output.image_mask,
                                 backbone_output.backbone_attention_mask).to(dtype=self.head.dtype)
            state_features=self.head.state_encoder(state,embodiment)
            if 'action' in action_input:
                # Preserve native RTC/inpainting and its validation. FP8 VLSA
                # still executes above; this branch uses the native DiT loop.
                return self.head.get_action_with_features(features,state_features,embodiment,
                                                         backbone_output,action_input,options)
            noise=torch.randn((1,40,132),dtype=self.head.dtype,device=features.device)
            actions=self.frontend.infer(state,initial_noise=noise)
            if not bool(torch.isfinite(actions).all()):
                raise RuntimeError('GR00T engine produced nonfinite actions')
            return BatchFeature(data={'action_pred':actions.to(self.head.dtype),
                                      'backbone_features':features,'state_features':state_features})


class _GrootEngineLoop:
    """Own the native loop so both precision modes retain its processing options."""

    def __init__(self, native_loop, frontend, runner, generator, device):
        self._native_loop = native_loop
        self._frontend, self._runner, self._generator = frontend, runner, generator
        self._device = device
        self._closed = False
        head = native_loop._policy.model.action_head
        self._head = head
        self._had_override = 'get_action' in head.__dict__
        self._original_override = head.__dict__.get('get_action')
        head.get_action = generator.get_action

    def __getattr__(self, name):
        return getattr(self._native_loop, name)

    @property
    def backend_stats(self):
        stats = dict(self._native_loop.backend_stats)
        # These are the native DiT driver's counters, which is not installed
        # underneath the FP8 generator. Report the actual VLSA replay counter.
        stats.pop('graph_captures', None)
        stats.pop('graph_replays', None)
        vlsa_graph = self._runner is not None and self._runner.graph is not None
        dit_graph = bool(self._frontend is not None and getattr(self._frontend, '_dit_graphs', None))
        model = getattr(getattr(self._native_loop, '_policy', None), 'model', None)
        backbone = getattr(model, 'backbone', None)
        graphs = getattr(backbone, '_instinctflash_fp8_graphs', ())
        stats.update(text_graphs=sum(g.graph is not None for g in graphs),
                     text_graph_rejections=[g.verdict for g in graphs if g.disabled],
                     backend='fp8_text_vlsa_native_vision_bf16_dit', precision='fp8',
                     captured=bool(vlsa_graph and dit_graph), vlsa_graph=vlsa_graph,
                     vlsa_replays=self._runner.replays if self._runner is not None else 0,
                     dit_graph=dit_graph, action_nfe=4,
                     fp8_recipe=getattr(self, '_fp8_recipe', None))
        return stats

    def reset(self, **conditioning):
        if self._closed:
            raise RuntimeError('GR00T engine loop is closed')
        return self._native_loop.reset(**conditioning)

    def predict(self, observation, *, executed_action=None):
        import torch
        if self._closed:
            raise RuntimeError('GR00T engine loop is closed')
        # Like the native Runtime, this stateless policy does not commit action
        # feedback. RTC inputs remain part of the native action-input contract.
        with torch.cuda.device(self._device):
            return self._native_loop.predict(observation)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._head is not None:
            if self._had_override:
                self._head.get_action = self._original_override
            else:
                del self._head.get_action
            self._head = None
        self._original_override = None
        model = getattr(getattr(self._native_loop, '_policy', None), 'model', None)
        backbone = getattr(model, 'backbone', None)
        for graph in getattr(backbone, '_instinctflash_fp8_graphs', ()):
            graph.close()
        try:
            self._native_loop.close()
        finally:
            self._frontend = self._runner = self._generator = None


def build_groot_engine_loop(checkpoint, *, device=None, action_steps=4):
    import torch
    from groot_n17_iwm.adapter import GR00TN17Adapter

    if action_steps != 4:
        raise ValueError('GR00T engine requires four action steps')
    dev = torch.device(device or 'cuda')
    if dev.type != 'cuda':
        raise ValueError('GR00T engine requires CUDA')
    from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
    from flash_rt.models.groot_n17.vlsa_runner import GrootN17VlsaRunner
    # Share native camera/LLM metadata caches, fast action decoding, embodiment,
    # thread configuration and their cleanup. No native DiT capture is requested;
    # the FP8 generator owns its own VLSA/DiT execution below.
    native_loop = GR00TN17Adapter().build_in_process(
        checkpoint, None, device=str(dev), nfe={'action': action_steps})
    try:
        with torch.cuda.device(dev):
            policy = native_loop._policy
            views = len(policy.modality_configs['video'].modality_keys)
            frontend = GrootN17TorchFrontendThor(
                str(native_loop._model_path), num_views=views,
                embodiment_tag=policy.embodiment_tag.value,
                device=str(dev), load_strided_fmha=False)
            runner = GrootN17VlsaRunner(frontend)
            generator = GrootActionGenerator(policy.model.action_head, frontend, runner)
            from .groot_fp8 import install_groot_backbone_fp8
            receipt = install_groot_backbone_fp8(policy.model.backbone)
            loop = _GrootEngineLoop(native_loop, frontend, runner, generator, dev)
            loop._fp8_recipe = receipt
            return loop
    except Exception:
        native_loop.close()
        raise
