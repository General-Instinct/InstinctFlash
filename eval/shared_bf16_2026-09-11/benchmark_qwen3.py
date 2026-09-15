"""Synthetic GR00T Qwen3-VL module screen, not a checkpoint/action benchmark."""
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import os
import torch
import transformers
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm, Qwen3VLTextMLP
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from instinctflash.backends import bf16_pointwise as kernels


def exact(a, b):
    return bool(torch.isfinite(a).all() and torch.isfinite(b).all()) and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g.replay, out, g


def timing(fn):
    for _ in range(5):
        fn()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(30)]
    for a, b in events:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in events)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    assert torch.cuda.get_device_capability() == (11, 0)
    def contention():
        ids = subprocess.check_output(['/usr/sbin/nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
        return [int(i) for i in ids if i.strip() and int(i) != os.getpid()]
    assert not contention(), 'Competing GPU process'
    torch.manual_seed(293)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'scope': __doc__, 'device': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'backend_sha256': hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(),
              'module_sources': {c.__name__: inspect.getsource(c) for c in (Qwen3VLTextMLP, Qwen3VLTextRMSNorm)},
              'weights': 'random; hidden=2048 intermediate=6144; not checkpoint weights', 'rows': []}
    with torch.inference_mode():
        cfg = Qwen3VLTextConfig(hidden_size=2048, intermediate_size=6144, hidden_act='silu')
        mlp = Qwen3VLTextMLP(cfg).cuda().bfloat16().eval()
        norm = Qwen3VLTextRMSNorm(2048).cuda().bfloat16().eval()
        norm.weight.copy_(torch.randn_like(norm.weight))
        table = kernels.silu_table('cuda')
        for tokens in (1, 128, 512):
            x = torch.randn(1, tokens, 2048, device='cuda', dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            for name, ref, candidate in (
                ('residual_norm', lambda: norm(x + residual), lambda: kernels.residual_norm(x, residual, norm.weight, norm.variance_epsilon, True)[0]),
                ('rmsnorm', lambda: norm(x), lambda: kernels.norm(x, norm.weight, norm.variance_epsilon, True)),
                ('full_mlp', lambda: mlp(x), lambda: mlp.down_proj(kernels.swiglu(mlp.gate_proj(x), mlp.up_proj(x), table))),
            ):
                matched = True
                for scale in (0.1, 1.0, 4.0):
                    x.copy_(torch.randn_like(x) * scale)
                    matched &= exact(ref(), candidate())
                rr, ro, rg = capture(ref)
                cr, co, cg = capture(candidate)
                for scale in (0.2, 2.0):
                    x.copy_(torch.randn_like(x) * scale)
                    rr(); cr()
                    matched &= exact(ro, co) and exact(ref(), co)
                for mode, a, b in (('eager', ref, candidate), ('graph', rr, cr)):
                    before = timing(a)
                    after = timing(b)
                    repeat = timing(a)
                    result['rows'].append(dict(operation=name, tokens=tokens, mode=mode,
                        finite_byte_equal=matched, reference_ms=before, candidate_ms=after,
                        reference_repeat_ms=repeat, speedup_vs_repeat=repeat / after))
                del rr, cr, ro, co, rg, cg
    result['competing_gpu_processes_at_end'] = contention()
    assert not result['competing_gpu_processes_at_end'], 'Competing GPU process'
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['rows'], indent=2))
    return 0 if all(r['finite_byte_equal'] for r in result['rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
