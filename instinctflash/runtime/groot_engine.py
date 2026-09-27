"""GR00T native processing with FP8 text projections and the GI FP8 action head."""


class _GrootEngineLoop:
    """Own the native loop so both precision modes retain its processing options."""

    def __init__(self, native_loop, generator, device):
        self._native_loop = native_loop
        self._generator = generator
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
        engine = self._generator.engine if self._generator is not None else None
        captured = bool(engine is not None and engine.graph_captured)
        model = getattr(getattr(self._native_loop, '_policy', None), 'model', None)
        backbone = getattr(model, 'backbone', None)
        graphs = getattr(backbone, '_instinctflash_fp8_graphs', ())
        stats.update(text_graphs=sum(g.graph is not None for g in graphs),
                     text_graph_rejections=[g.verdict for g in graphs if g.disabled],
                     backend='fp8_text_gi_action_head_native_vision', precision='fp8',
                     captured=captured, action_head_graphs=len(engine._graphs.graphs) if engine else 0,
                     action_head_replays=engine.replays if engine else 0, action_nfe=4,
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
            self._generator = None


def build_groot_engine_loop(checkpoint, *, device=None, action_steps=4):
    import torch
    from groot_n17_iwm.adapter import GR00TN17Adapter

    if action_steps != 4:
        raise ValueError('GR00T engine requires four action steps')
    dev = torch.device(device or 'cuda')
    if dev.type != 'cuda':
        raise ValueError('GR00T engine requires CUDA')
    from instinctflash.engines.groot_n17.frontend import GiGrootActionGenerator
    # Share native camera/LLM metadata caches, fast action decoding, embodiment,
    # thread configuration and their cleanup. No native DiT capture is requested;
    # the GI generator owns VLSA/DiT execution below.
    native_loop = GR00TN17Adapter().build_in_process(
        checkpoint, None, device=str(dev), nfe={'action': action_steps})
    try:
        with torch.cuda.device(dev):
            policy = native_loop._policy
            generator = GiGrootActionGenerator.from_checkpoint(
                policy.model.action_head, str(native_loop._model_path), policy.embodiment_tag.value, dev)
            from .groot_fp8 import install_groot_backbone_fp8
            receipt = install_groot_backbone_fp8(policy.model.backbone)
            loop = _GrootEngineLoop(native_loop, generator, dev)
            loop._fp8_recipe = receipt
            return loop
    except Exception:
        native_loop.close()
        raise
