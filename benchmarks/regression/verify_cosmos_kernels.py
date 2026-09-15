"""Standalone Thor primitive qualification; no pytest installation required."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def verify_pointwise(kernels):
    torch.manual_seed(293)
    with torch.inference_mode():
        for (n, d) in [(19, 128), (152, 2048), (3093, 4096)]:
            x = torch.randn(n, d, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(d, device='cuda', dtype=torch.bfloat16)
            scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-06)
            for first in [False, True]:
                ref = w * (x.float() * scale).to(x.dtype) if first else (w.float() * (x.float() * scale)).to(x.dtype)
                assert kernels._exact(ref, kernels.norm(x, w, 1e-06, first))
            assert kernels._exact(torch.relu(x).square(), kernels.relu2(x))
        for (n, hq, hk, d, r) in [(19, 16, 16, 128, 128), (31, 32, 8, 128, 64), (3093, 16, 16, 128, 128)]:
            q = torch.randn(n, hq, d, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(n, hk, d, device='cuda', dtype=torch.bfloat16)
            c = torch.randn(n, r, device='cuda', dtype=torch.bfloat16)
            s = torch.randn_like(c)

            def eager(x):
                a = x[..., :r]
                rotated = torch.cat((-a[..., r // 2:], a[..., :r // 2]), -1)
                return torch.cat((a * c[:, None, :] + rotated * s[:, None, :], x[..., r:]), -1)
            assert kernels._exact((eager(q), eager(k)), kernels.rope(q, k, c, s))

def verify_swiglu(kernels):
    with torch.inference_mode():
        bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        values = values[torch.isfinite(values)].contiguous()
        up = torch.ones_like(values)
        reference = torch.nn.functional.silu(values) * up
        candidate = kernels.swiglu(values, up, kernels.silu_table(values.device))
        assert kernels._exact(reference, candidate)


def verify_pointwise_graph_replay(kernels):
    """Captured shared SwiGLU must observe new values, not warmup activations."""
    with torch.inference_mode():
        gate = torch.randn(19, 256, device='cuda', dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        table = kernels.silu_table(gate.device)
        for _ in range(3):
            kernels.swiglu(gate, up, table)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = kernels.swiglu(gate, up, table)
        for scale in (0.1, 3.0, 0.0):
            gate.copy_(torch.randn_like(gate) * scale)
            up.copy_(torch.randn_like(up))
            expected = torch.nn.functional.silu(gate) * up
            graph.replay()
            assert kernels._exact(expected, out)
            held = out.clone()
            kernels.swiglu(torch.ones_like(gate), up, table)
            assert kernels._exact(held, out)


def verify_residual_norm(kernels):
    from instinctflash.backends.bf16_pointwise import residual_norm
    with torch.inference_mode():
        for rows, width in ((1, 768), (19, 2048), (512, 2048)):
            x = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            w = torch.randn(width, device='cuda', dtype=torch.bfloat16)
            for first in (False, True):
                for _ in range(3):
                    residual_norm(x, residual, w, 1e-6, first)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out, added = residual_norm(x, residual, w, 1e-6, first)
                for cancellation in (False, True):
                    x.copy_(torch.randn_like(x))
                    residual.copy_(-x if cancellation else torch.randn_like(x))
                    expected_sum = x + residual
                    scaled = expected_sum.float() * torch.rsqrt(expected_sum.float().square().mean(-1, keepdim=True) + 1e-6)
                    expected = w * scaled.bfloat16() if first else (w.float() * scaled).bfloat16()
                    graph.replay()
                    assert kernels._exact((expected, expected_sum), (out, added))
                    retained = out.clone(), added.clone()
                    residual_norm(x, residual, w, 1e-6, first)
                    assert kernels._exact(retained, (out, added))


def verify_graphs(kernels):
    from cosmos3_iwm.thor_graphs import LayerGraph
    with torch.inference_mode():
        module = torch.nn.Linear(16, 16).cuda().bfloat16()
        original = module.forward
        stats = {'captures': 0, 'checks': 0, 'replays': 0, 'rejected': []}
        graph = LayerGraph(original, stats, 'linear')
        for rows in (17, 17, 17, 19, 19, 19):
            x = torch.randn(rows, 16, device='cuda', dtype=torch.bfloat16)
            assert kernels._exact(original(x), graph(x))
        # Replacing a parameter must not replay a graph bound to old weights.
        module.weight = torch.nn.Parameter(torch.ones_like(module.weight))
        x = torch.randn(17, 16, device='cuda', dtype=torch.bfloat16)
        assert kernels._exact(original(x), graph(x))
        assert stats['captures'] == 3 and stats['replays'] >= 2 and not stats['rejected']
        # Exercise shared-pool graphs in changing order and retain earlier outputs.
        # This is the reuse pattern adapted from GR00T's flow graph cache.
        pool = torch.cuda.graph_pool_handle()
        modules = [torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(),
                                       torch.nn.Linear(64, 16)).cuda().bfloat16() for _ in range(2)]
        shared = [LayerGraph(m.forward, stats, str(i), pool=pool) for i, m in enumerate(modules)]
        held = []
        for i, rows in [(0, 17), (1, 19), (1, 17), (0, 19), (1, 19), (0, 17), (0, 19), (1, 17),
                        (0, 19), (1, 19), (1, 17), (0, 17)]:
            x = torch.randn(rows, 16, device='cuda', dtype=torch.bfloat16)
            reference = modules[i](x)
            out = shared[i](x)
            assert kernels._exact(reference, out)
            # A replay can overwrite its own borrowed output, but never another entry's.
            for owner, previous, expected in held:
                if owner != (i, rows):
                    assert kernels._exact(previous, expected)
            held = [(owner, previous, expected) for owner, previous, expected in held if owner != (i, rows)]
            held.append(((i, rows), out, reference.clone()))
        assert not stats['rejected']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (11, 0)
    from cosmos3_iwm import exact_pointwise as kernels
    from cosmos3_iwm import thor_graphs
    from instinctflash.backends import bf16_pointwise
    result = {'shared_pointwise_sha256': hashlib.sha256(Path(bf16_pointwise.__file__).read_bytes()).hexdigest(), 'graph_source_sha256': hashlib.sha256(Path(thor_graphs.__file__).read_bytes()).hexdigest(), 'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'source_sha256': hashlib.sha256(Path(kernels.__file__).read_bytes()).hexdigest(), 'checks': {}}
    for name, function in [('norm_relu2_rope', verify_pointwise), ('residual_norm_rounding_graph_ownership', verify_residual_norm), ('all_finite_bf16_swiglu', verify_swiglu), ('pointwise_changed_input_graph', verify_pointwise_graph_replay), ('changed_input_shape_weight_graphs', verify_graphs)]:
        try:
            function(kernels)
            result['checks'][name] = 'PASS'
        except Exception as error:
            result['checks'][name] = repr(error)
    result['status'] = 'PASS' if all(x == 'PASS' for x in result['checks'].values()) else 'FAIL'
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
