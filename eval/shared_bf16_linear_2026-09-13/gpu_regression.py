"""Real-operand shared-kernel tests: streams, compile, fallbacks and ABI rejection."""
import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from instinctflash.backends.bf16_linear import BF16LinearActivation, _library
from instinctflash.planners.planner import Tier

root = Path(__file__).resolve().parent
out = root / 'gpu_regression.json'
assert not out.exists()
report = dict(ok=False, checks=[], task_quality_certified=False)
def record(name):
    report['checks'].append(name)
    out.write_text(json.dumps(report, indent=2)+'\n')
payload = torch.load('/home/guanming/ifl_eval/edge_mlp_tuning_20260913_v2/operands.pt',
                     map_location='cuda', weights_only=True)
plan = SimpleNamespace(tier_ceiling=Tier.NUMERIC)
torch.backends.cuda.matmul.allow_tf32 = False
with torch.inference_mode():
    for layer, state in payload.items():
        linear = torch.nn.Linear(2048, 9216, bias=False, device='meta', dtype=torch.bfloat16)
        linear.weight = torch.nn.Parameter(state['up_weight'], requires_grad=False)
        fused = BF16LinearActivation(linear, lambda x: x.relu().square(),
                                    activation_name='relu2', plan=plan)
        compiled = torch.compile(fused, fullgraph=True, dynamic=False,
                                 options={'triton.cudagraphs': False})
        for i, sample in enumerate(state['samples']):
            x = sample['x']
            expected = linear(x).relu().square()
            assert torch.equal(fused(x), expected)
            assert torch.equal(compiled(x), expected)
            record(f'{layer}/{i}: eager+fullgraph equal')
            # Allocations and native launches must follow each caller stream.
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            outputs = []
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs.append(fused(x))
            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)
            assert all(torch.equal(y, expected) for y in outputs)
            record(f'{layer}/{i}: two independent CUDA streams equal')
            short = x[:3]
            assert torch.equal(fused(short), linear(short).relu().square())
            record(f'{layer}/{i}: short-shape original fallback equal')
        static = state['samples'][0]['x'].clone()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                fused(static)
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fused(static)
        static.copy_(state['samples'][1]['x'])
        graph.replay()
        assert torch.equal(graph_output, linear(static).relu().square())
        record(f'{layer}: CUDA graph replay with changed input equal')
        del graph, graph_output
        fn = _library(fused.library).instinctflash_bf16_linear_relu2
        # Invalid native input fails before dereferencing pointers or launching.
        assert fn(None, None, None, 1, 1, 1, None, 0, None) == -3
        record(f'{layer}: raw ABI rejects invalid arguments')
report['ok'] = True
out.write_text(json.dumps(report, indent=2)+'\n')
