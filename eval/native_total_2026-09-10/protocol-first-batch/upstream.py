"""Direct upstream policy constructors. Shared wrappers only translate public I/O.

No adapter build/install, graph, hoist, fast decode or GPU preprocessing is called.
The checkpoint declaration is resolved by Runtime without loading its backend.
"""
import os
import sys
from types import SimpleNamespace


def build(family, declaration):
    checkpoint = declaration._checkpoint
    extra = dict(checkpoint.execution.extra or {})
    steps = dict(checkpoint.execution.nfe or {})
    dev = 'cuda:0'
    if family == 'pi05':
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from pi05_iwm.adapter import _Pi05Loop
        repo = extra['base_weights']
        config = PI05Config.from_pretrained(repo)
        # Explicit eager PyTorch reference; dtype is kept from the checkpoint.
        config.compile_model = False
        policy = PI05Policy.from_pretrained(repo, config=config).eval().to(dev)
        policy.config.num_inference_steps = int(steps.get('action', 10))
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=repo,
            preprocessor_overrides={'device_processor': {'device': dev}},
            postprocessor_overrides={'device_processor': {'device': dev}})
        loop = _Pi05Loop(policy, pre, post, dev)
    elif family == 'vla4':
        from lingbot_vla_iwm.adapter import (
            _source_root, _resolve_model_path, _resolve_norm_stats, _LingBotVLA4BLoop)
        root = _source_root()
        sys.path.insert(0, str(root))
        os.environ['QWEN25_PATH'] = os.environ.get('QWEN25_PATH') or str(extra.get('tokenizer_repo') or 'Qwen/Qwen2.5-VL-3B-Instruct')
        from deploy.lingbot_vla_policy import LingbotVLAServer
        server = LingbotVLAServer(str(_resolve_model_path(checkpoint)),
            use_length=int(extra.get('use_length') or 25),
            robot_norm_path=str(_resolve_norm_stats(checkpoint, root)),
            num_denoising_step=int(steps.get('action', 10)))
        loop = _LingBotVLA4BLoop(server, root, robot=str(extra.get('robot') or 'robotwin'))
    elif family == 'groot':
        from groot_n17_iwm.adapter import (
            _source_root, _resolve_model_path, DEFAULT_EMBODIMENT, _GR00TN17Loop)
        sys.path.insert(0, str(_source_root()))
        import transformers.tokenization_utils_base as tub
        # Same offline Qwen tokenizer compatibility workaround as the adapter.
        if hasattr(tub.PreTrainedTokenizerBase, '_patch_mistral_regex'):
            tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, tokenizer, *args, **kwargs: tokenizer)
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        path = _resolve_model_path(checkpoint)
        policy = Gr00tPolicy(embodiment_tag=str(extra.get('embodiment_tag') or DEFAULT_EMBODIMENT), model_path=str(path), device=dev, strict=True)
        nfe = int(steps.get('action', 4))
        policy.model.action_head.num_inference_timesteps = nfe
        if hasattr(policy.model, 'config'):
            policy.model.config.num_inference_timesteps = nfe
        loop = _GR00TN17Loop(policy, model_path=path, action_nfe=nfe)
    else:
        raise ValueError(f'No audited upstream constructor for {family}')
    return Reference(declaration, loop)


class Reference:
    def __init__(self, declaration, loop):
        self._declaration = declaration
        self._checkpoint = declaration._checkpoint
        self._backend = SimpleNamespace(_impl=loop)
        self.plan = declaration.plan.without(*[r.name for r in declaration.plan.results])
        self.execution_policy = {'reference': 'upstream eager native policy; shared I/O translation only'}

    def reset(self, **kwargs):
        return self._backend._impl.reset(**kwargs)

    def predict(self, observation, **kwargs):
        assert not kwargs
        return self._backend._impl.predict(observation)

    def close(self):
        close = getattr(self._backend._impl, 'close', None)
        if close:
            close()
        self._declaration.close()
