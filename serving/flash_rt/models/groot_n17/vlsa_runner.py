"""Live four-layer FP8 VLSA stage; input is the current language-model output.

This component does not supply the preceding vision/language model or native
robot processors. It must not be advertised as a complete FP8 Runtime on its own.
"""
import torch
from . import calibration as cal
from . import pipeline_thor as pipeline


class GrootN17VlsaRunner:
    def __init__(self, frontend, *, use_cuda_graph=True):
        import flash_rt.flash_rt_kernels as fvk
        self.frontend = frontend
        self.fvk = fvk
        self.gemm = fvk.GemmRunner()
        self.ctx = fvk.FvkContext()
        self.use_cuda_graph = use_cuda_graph
        self.sequence = None
        self.graph = None
        self.replays = 0
        for name in ('q','k','v','o','fc1','fc2'):
            weights = getattr(frontend,f'_vlsa_{name}_w')
            if len(weights) != 4 or any(w.dtype != torch.float8_e4m3fn for w in weights):
                raise ValueError('VLSA runner requires four layers of actual E4M3 weights')

    def _prepare(self, sample):
        from flash_rt.hardware.thor.attn_backend_groot_n17 import (
            ThorGrootN17AttnBackend, make_groot_n17_attention_spec,
        )
        fe = self.frontend
        self.graph = None
        S = sample.shape[1]
        padded = (S+7)//8*8
        def alloc(shape,dtype=torch.float16):
            return torch.empty(shape,dtype=dtype,device=fe.device)
        self.input = alloc((S,2048))
        self.buffers = {key:alloc((S,2048)) for key in ('h','xn','o_proj_out','Q','K','V','O')}
        self.buffers.update(xn_fp8=alloc((S,2048),torch.float8_e4m3fn),
                            fc1_out=alloc((S,8192)),
                            fc1_fp8=alloc((S,8192),torch.float8_e4m3fn),
                            logits=alloc((32,padded,padded)))
        # Only vl_self_attn is executed by this runner. The multi-site backend
        # requires descriptors for every family site; unused sites share storage.
        slots={name:self.buffers[name].data_ptr() for name in ('Q','K','V','O','logits')}
        slots.update(ctx=self.ctx,scale=1/8)
        cross={key:value for key,value in slots.items() if key not in ('K','V')}
        cross.update(K_layers=[slots['K']]*16,V_layers=[slots['V']]*16)
        spec=make_groot_n17_attention_spec(num_views=fe.num_views,llm_seq_max=padded,
            vl_self_attn_seq_max=padded,sa=41,s_kv_text=1,s_kv_image=1)
        self.attn=ThorGrootN17AttnBackend(spec,vit_slots={'qkv':slots['Q'],'O':slots['O'],'D':1024},
            llm_slots=slots,vl_self_attn_slots=slots,dit_self_slots=slots,dit_cross_slots=cross)
        calibration = cal.calibrate_vlsa(fe,sample.float())
        self.scales = {}
        for name in ('qkv','o','fc1','fc2'):
            values=calibration[f'vlsa_act_{name}']
            if len(values)!=4 or any(not (0 < float(v) < float('inf')) for v in values):
                raise RuntimeError('invalid VLSA calibration')
            self.scales[f'act_{name}']=[cal.amax_to_dev_scale(v,device=fe.device) for v in values]
        self.weights={}
        for name in ('norm1_w','norm1_b','norm3_w','norm3_b',
                     'q_w','q_b','k_w','k_b','v_w','v_b','o_w','o_b',
                     'fc1_w','fc1_b','fc2_w','fc2_b'):
            self.weights[name]=[w.data_ptr() for w in getattr(fe,f'_vlsa_{name}')]
        for index,name in enumerate(('q','k','v','o','fc1','fc2')):
            activation='qkv' if name in ('q','k','v') else name
            self.weights[f'alpha_{name}']=[cal.alpha(calibration[f'vlsa_act_{activation}'][layer],
                fe._vlsa_alpha[layer*6+index]) for layer in range(4)]
        self.sequence = S
        self.calibration = {k:v for k,v in calibration.items() if k!='backbone_features'}

    def _run(self, stream):
        fe = self.frontend
        pipeline.vlln_forward(self.gemm,self.fvk,
            {'x':self.input.data_ptr(),'out':self.buffers['h'].data_ptr()},
            {'vlln_w':fe._vlln_w.data_ptr(),'vlln_b':fe._vlln_b.data_ptr()},
            {'S':self.sequence,'D':2048},stream=stream)
        pipeline.vl_self_attn_forward(self.gemm,self.fvk,
            {k:v.data_ptr() for k,v in self.buffers.items()},self.weights,
            {'T':self.sequence,'D':2048,'NH':32,'HD':64,'ff_inner':8192},
            {k:[v.data_ptr() for v in values] for k,values in self.scales.items()},
            attn=self.attn,stream=stream)

    @torch.no_grad()
    def __call__(self, llm_final, visual_mask, attention_mask=None):
        sample=torch.as_tensor(llm_final,device=self.frontend.device)
        if sample.ndim!=3 or sample.shape[0]!=1 or sample.shape[2]!=2048 or sample.shape[1]<2:
            raise ValueError('VLSA expects current LLM output [1,tokens,2048]')
        sample=sample.to(torch.float16)
        mask=torch.as_tensor(visual_mask,device=sample.device)
        if mask.dtype!=torch.bool or mask.numel()!=sample.shape[1] or not bool(mask.any()) or not bool((~mask).any()):
            raise ValueError('VLSA requires a boolean text/image mask for the current tokens')
        if not bool(torch.isfinite(sample).all()):
            raise ValueError('VLSA input is nonfinite in engine precision')
        valid = (torch.ones(sample.shape[1],dtype=torch.bool,device=sample.device)
                 if attention_mask is None else torch.as_tensor(attention_mask,device=sample.device).reshape(-1).bool())
        if valid.numel()!=sample.shape[1] or not bool((mask.reshape(-1)&valid).any()) or not bool((~mask.reshape(-1)&valid).any()):
            raise ValueError('DiT requires active image and text tokens')
        with torch.cuda.stream(torch.cuda.default_stream(sample.device)):
            if self.sequence!=sample.shape[1]:
                self._prepare(sample)
            self.input.copy_(sample.squeeze(0))
            if self.graph is None and self.use_cuda_graph:
                for _ in range(3):self._run(0)
                torch.cuda.synchronize(sample.device)
                stream=torch.cuda.Stream(device=sample.device)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream),torch.cuda.graph(graph,stream=stream):
                    self._run(stream.cuda_stream)
                torch.cuda.synchronize(sample.device)
                self.graph=graph
            if self.graph is not None:
                self.graph.replay();self.replays+=1
            else:
                self._run(0)
            result=self.buffers['h'].unsqueeze(0)
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError('FP8 VLSA returned nonfinite features')
            # Native VLSA processes all tokens; the following DiT applies the
            # backbone mask. Gather only after VLSA to preserve that ordering.
            self.frontend.update_backbone_features(result[:,valid],mask.reshape(-1)[valid])
            return result.clone()
