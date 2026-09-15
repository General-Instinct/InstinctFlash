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
    steps = {**dict(checkpoint.execution.nfe or {}), **dict(declaration._requested_nfe or {})}
    dev = 'cuda:0'
    if family == 'pi05':
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from pi05_iwm.adapter import _Pi05Loop
        repo = extra['base_weights']
        config = PI05Config.from_pretrained(repo)
        # Explicit eager PyTorch reference; dtype is kept from the checkpoint.
        config.compile_model = True
        config.num_inference_steps = int(steps.get("action",10))
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
    elif family == 'vla2':
        from lingbot_vla_v2_iwm.adapter import (_source_root, _resolve_model_path,
            _configure_thor_ptxas, _LingBotVLAV2Loop)
        import torch
        _configure_thor_ptxas(torch.cuda.get_device_capability())
        root = _source_root()
        sys.path.insert(0, str(root))
        os.environ['QWEN3VL_PATH'] = os.environ.get('QWEN3VL_PATH') or str(extra.get('tokenizer_repo') or 'Qwen/Qwen3-VL-4B-Instruct')
        import transformers.tokenization_utils_base as tub
        if hasattr(tub.PreTrainedTokenizerBase, '_patch_mistral_regex'):
            tub.PreTrainedTokenizerBase._patch_mistral_regex = classmethod(lambda cls, tokenizer, *args, **kwargs: tokenizer)
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
        server = LingbotVLAv2Server(str(_resolve_model_path(checkpoint)), use_length=50,
            chunk_ret=True, use_bf16=True, use_fp32=False, use_compile=False)
        server.vla.model.config.num_steps = int(steps.get('action', 10))
        loop = _LingBotVLAV2Loop(server, root, robot=str(extra.get('robot') or 'robotwin'))
    elif family in ('edge', 'nano'):
        from cosmos3_iwm.adapter import _resolve_model_path
        from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
        from instinctflash.runtime.cosmos_droid import CosmosDROIDLoop
        class EagerService(RobolabPolicyService):
            def _build_setup_args(self, args):
                return super()._build_setup_args(args).model_copy(update={'guardrails': False, 'use_torch_compile': False})
        service = EagerService(RobolabServerArgs(
            checkpoint_path=str(_resolve_model_path(checkpoint)),
            **{k: extra[k] for k in ('domain_name', 'action_chunk_size', 'conditioning_fps',
                'action_dim', 'image_height', 'image_width', 'format_prompt_as_json')},
            num_steps=int(steps['action']), guidance=3.0,
            shift=float(extra.get('shift', 5.0)), seed=int(extra.get('seed', 0))))
        loop = CosmosDROIDLoop(service)
    elif family == 'va':
        from copy import deepcopy
        from instinctflash.adapters.lingbot_va import (LingBotVA, _ControlLoop,
            resolve_observation_geometry, apply_declared_guidance)
        from instinctflash.runtime.lingbot_install import import_lingbot_server
        adapter = LingBotVA()
        composed = adapter.materialize(checkpoint)  # symlinks to declared weights only
        os.environ['LINGBOT_CKPT'] = composed
        server_module = import_lingbot_server(adapter.lingbot_root)
        cfg = deepcopy(server_module.VA_CONFIGS[extra.get('va_config') or 'robotwin'])
        geometry, _ = resolve_observation_geometry(checkpoint.execution, va_configs=server_module.VA_CONFIGS)
        for key, value in geometry.items():
            setattr(cfg, key, value)
        for key in ('wan22_pretrained_model_name_or_path', 'pretrained_model_name_or_path'):
            if hasattr(cfg, key):
                setattr(cfg, key, composed)
        import tempfile
        cfg.save_root = tempfile.mkdtemp(prefix='ifl-upstream-va-')
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29531')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        server_module.init_distributed(1, 0, 0)
        cfg.rank = cfg.local_rank = 0
        cfg.world_size = 1
        cfg.num_inference_steps = int(steps['video'])
        cfg.action_num_inference_steps = int(steps['action'])
        apply_declared_guidance(cfg, checkpoint.execution.guidance)
        server = server_module.VA_Server(cfg)
        loop = _ControlLoop(server, tuple(cfg.obs_cam_keys), frame_chunk_size=cfg.frame_chunk_size)
    elif family == 'dreamzero':
        from dreamzero_iwm.adapter import _source_root, _resolve_model_path, _DreamZeroLoop, _head_declaration
        sys.path.insert(0, str(_source_root()))
        from eval_utils.serve_dreamzero_wan22 import DreamZeroWan225BPolicy, _get_expected_video_resolution, _maybe_init_distributed
        from groot.vla.data.schema import EmbodimentTag
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from torch.distributed.device_mesh import init_device_mesh
        _maybe_init_distributed()
        mesh = init_device_mesh('cuda', mesh_shape=(1,), mesh_dim_names=('ip',))
        tag = str(extra.get('embodiment_tag') or 'oxe_droid')
        path = _resolve_model_path(checkpoint)
        view = receipt = None
        if os.environ.get('IFL_BENCH_DREAMZERO_FULL_CHECKPOINT') == '1':
            # Upstream's own full-checkpoint option; keep native FP32 initialization.
            # Coverage validation proves the omitted base DiT is overwritten in full.
            from instinctflash.runtime.dreamzero_checkpoint import prepare_full_checkpoint, DIT_TARGET
            from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
            import json
            view, receipt = prepare_full_checkpoint(path, CausalWanModel, direct_bf16=False)
            path = type(path)(view.name)
            config = json.loads((path / 'config.json').read_text())
            assert config['action_head_cfg']['config']['diffusion_model_cfg']['_target_'] == DIT_TARGET
            assert receipt['direct_bf16_dit'] is False
            receipt = {k: v for k, v in receipt.items() if k != 'expected_shapes'}
            receipt['reference'] = 'Native full-checkpoint flag; original initialization and original forward implementation'
        try:
            policy = GrootSimPolicy(embodiment_tag=EmbodimentTag(tag),
                model_path=str(path), tokenizer_path_override=None,
                device='cuda', device_mesh=mesh)
            height, width = _get_expected_video_resolution(policy)
            wrapper = DreamZeroWan225BPolicy(groot_policy=policy, image_height=height,
                image_width=width, embodiment_tag=tag)
            head = policy.trained_model.action_head
            loop = _DreamZeroLoop(wrapper, dynamic_cache=bool(head.dynamic_cache_schedule),
                build_declaration=_head_declaration(head), checkpoint_view=view, loading_receipt=receipt)
        except Exception:
            if view is not None:
                view.cleanup()
            raise
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
        action = self._backend._impl.predict(observation)
        commit = getattr(self._backend._impl, 'commit', None)
        if commit:
            commit(dict(observation), kwargs.get('executed_action', action['action']))
        return action

    def close(self):
        close = getattr(self._backend._impl, 'close', None)
        if close:
            close()
        self._declaration.close()
