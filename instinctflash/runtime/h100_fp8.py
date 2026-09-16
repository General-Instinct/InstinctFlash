"""H100 opt-in E4M3 attention projections, before native CUDA graph capture.

This executor is distinct from the Thor fused kernels. Native processors, attention,
MoE routing, MLPs, schedules and control history are retained. No quality certificate
is implied by successful construction or graph self-checks.
"""
from __future__ import annotations
import torch
from torch import nn
from .torch_fp8_linear import ThorFP8Linear

class H100FP8Linear(ThorFP8Linear):
    @property
    def weight(self):
        # Some upstream attention implementations inspect dtype/device. An actual
        # direct-weight matmul must fail rather than silently avoid quantization.
        return torch.empty(0, device=self.weight_fp8.device, dtype=torch.bfloat16)

    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.bfloat16)
    def forward(self, x):
        return super().forward(x)


def maybe_install_h100_fp8(model, plan, family):
    requested = any(r.name == 'engine_offload' and r.applies and
                    r.params.get('executor') == 'h100_torch_fp8'
                    for r in getattr(plan, 'results', ()))
    if not requested:
        return
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('H100 FP8 recipe requires SM90')
    if hasattr(model, '_h100_fp8_recipe'):
        raise ValueError('H100 FP8 already installed')
    names = {'q_proj','k_proj','v_proj','to_q','to_k','to_v','q','k','v'}
    targets=[]
    for path,parent in model.named_modules():
        if 'attention' not in type(parent).__name__.lower() and 'attn' not in type(parent).__name__.lower():
            continue
        for name,source in parent.named_children():
            if name not in names or type(source) is not nn.Linear:
                continue
            # FP32 projections remain FP32; this recipe does not silently recast them.
            if source.weight.dtype != torch.bfloat16:
                continue
            if source.in_features % 16 or source.out_features % 16:
                continue
            if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
                raise ValueError(f'Cannot replace hooked projection {path}.{name}')
            targets.append((path,parent,name,source))
    if not targets:
        raise ValueError(f'No eligible native BF16 attention projections for {family}')
    packed=[H100FP8Linear(source) for _,_,_,source in targets]
    recipe={'executor':'h100_torch_fp8','precision':'fp8','family':family,
            'recipe':ThorFP8Linear.recipe,'scope':'BF16 attention Q/K/V only; no MLP, MoE expert, norm, processor or schedule change',
            'projections':[{'path':f'{path}.{name}','parent_type':type(parent).__name__,
                            'in_features':s.in_features,'out_features':s.out_features}
                           for path,parent,name,s in targets]}
    for (_,parent,name,_),replacement in zip(targets,packed):setattr(parent,name,replacement)
    model._h100_fp8_recipe=recipe

class H100Loop:
    def __init__(self,loop,recipe,*,executor='h100_torch_fp8'):
        self.inner,self.recipe,self.executor=loop,recipe,executor
    def reset(self,**kw):return self.inner.reset(**kw)
    def predict(self,obs,*,executed_action=None):
        validate=getattr(self.inner,'validate_executed_action',None)
        if validate is not None:validate(executed_action)
        commit=getattr(self.inner,'commit',None)
        if commit is not None:
            out=self.inner.predict(dict(obs))
            action=out.get('action') if isinstance(out,dict) else out
            commit(dict(obs), action if executed_action is None else executed_action)
            return out
        if executed_action is None:return self.inner.predict(obs)
        import inspect
        if 'executed_action' not in inspect.signature(self.inner.predict).parameters:
            raise ValueError('This policy does not accept executed-action feedback')
        return self.inner.predict(obs,executed_action=executed_action)
    def close(self):return self.inner.close()
    @property
    def backend_stats(self):
        stats=getattr(self.inner,'backend_stats',{})
        if callable(stats):stats=stats()
        return {'native_backend':stats,'fp8_recipe':self.recipe}
    def declaration(self):return {'precision':'fp8','executor':self.executor,'recipe':self.recipe}


def build_h100_loop(adapter,checkpoint,plan,*,device=None,nfe=None,step_cache=None):
    family=checkpoint.execution.backbone
    if family == 'wan_va':
        from dataclasses import replace
        for i,result in enumerate(plan.results):
            if result.name == 'cfg_branch_elision' and result.applies:
                plan.results[i]=replace(result,applies=False,reason='H100 FP8 retains native CFG execution; no CFG elision installer')
    if family in ('cosmos3_policy','dreamzero'):
        build_kw={'plan':plan} if family=='dreamzero' else {}
        if family=='dreamzero' and step_cache is not None:
            build_kw['step_cache']=step_cache
        loop=adapter.build_fp8(checkpoint,device=device,nfe=nfe,**build_kw)
        try:
            stats=loop.backend_stats
            if callable(stats):stats=stats()
            recipe=stats['fp8_recipe']
            if not recipe.get('projections'):raise ValueError('No actual FP8 projections')
            return H100Loop(loop,recipe)
        except Exception:
            loop.close();raise
    loop=adapter.build_in_process(checkpoint,plan,device=device,nfe=nfe)
    try:
        if family=='pi05':model=loop._p.model
        elif family=='groot_n17':model=loop._policy.model
        elif family in ('lingbot_vla','lingbot_vla_v2'):model=loop._server.vla.model
        elif family=='wan_va':model=loop._server.transformer
        else:raise ValueError(f'No H100 recipe for {family}')
        recipe=model._h100_fp8_recipe
        if not recipe['projections']:raise ValueError('FP8 requested but not installed')
        return H100Loop(loop,recipe)
    except Exception:
        loop.close();raise
