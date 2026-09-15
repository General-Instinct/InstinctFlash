import json
from pathlib import Path
import torch
from instinctflash.runtime.torch_fp8_linear import ThorFP8Linear

torch.manual_seed(29)
torch.backends.cuda.matmul.allow_tf32 = False
records = []
for bias in (False, True):
    native = torch.nn.Linear(2048, 1024, bias=bias, device='cuda', dtype=torch.bfloat16).eval()
    fp8 = ThorFP8Linear(native)
    for shape in [(1, 2048), (128, 2048), (2, 3, 2048), (0, 2048), (2048,)]:
        x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            ref, y = native(x), fp8(x)
        assert y.shape == ref.shape and torch.isfinite(y).all()
        rmse = ((y.float()-ref.float()).square().mean().sqrt() / ref.float().square().mean().sqrt()).item() if y.numel() else None
        records.append(dict(bias=bias, shape=shape, relative_rmse=rmse))
    x = torch.randn(4, 2, 2048, device='cuda', dtype=torch.bfloat16).transpose(0, 1)
    assert not x.is_contiguous()
    assert torch.equal(fp8(x), fp8(x.contiguous()))
    zero = torch.zeros(4, 2048, device='cuda', dtype=torch.bfloat16)
    target = native.bias.expand(4, -1) if bias else torch.zeros(4, 1024, device='cuda', dtype=torch.bfloat16)
    assert torch.equal(fp8(zero), target)
    # Capture on a warmed side stream; change values at the same graph addresses.
    x = torch.randn(4, 2048, device='cuda', dtype=torch.bfloat16)
    saved = x.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fp8(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fp8(x)
    graph.replay()
    first = out.clone()
    x.mul_(-2)
    graph.replay()
    changed = out.clone()
    assert not torch.equal(first, changed)
    assert torch.equal(changed, fp8(x))
    x.copy_(saved)
    graph.replay()
    assert torch.equal(first, out)
    with torch.no_grad():
        native.weight.zero_()
    zero_weight = ThorFP8Linear(native)
    assert torch.equal(zero_weight(x), target)
    fp8.to(dtype=torch.bfloat16)
    try:
        fp8(x)
    except RuntimeError as e:
        assert 'cast after construction' in str(e)
    else:
        raise AssertionError('cast packed weights were silently accepted')
torch.cuda.synchronize()
report = dict(ok=True, torch_version=torch.__version__, device=torch.cuda.get_device_name(), recipe=ThorFP8Linear.recipe, cases=records, graph_changed_and_restored=True, noncontiguous=True, zero_input_and_weight=True, packed_cast_refused=True, scope='Thor operator execution only; no Cosmos model speed or task quality certificate')
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
