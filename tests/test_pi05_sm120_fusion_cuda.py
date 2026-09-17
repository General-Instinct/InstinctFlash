"""Real SM120 operator equivalence; the reference is the existing CUDA sequence."""
import sys
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("requires a real SM120 GPU", allow_module_level=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
from flash_rt import flash_rt_kernels as old
from flash_rt import flash_rt_pi05_sm120_fusion as fusion


def identical(a, b):
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), int((a.view(torch.uint8) != b.view(torch.uint8)).sum())


def values(shape, pattern):
    n = 1
    for dimension in shape: n *= dimension
    if pattern == "words":
        return (torch.arange(n, device="cuda", dtype=torch.int32) % 65536).to(torch.uint16).view(torch.bfloat16).reshape(shape)
    if pattern == "alternating":
        return (torch.arange(n, device="cuda") % 2 * 2 - 1).to(torch.bfloat16).reshape(shape) * 16
    if pattern == "zero":
        return torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * float(pattern)


@pytest.mark.parametrize("pattern", ["0.01", "1", "16", "zero", "words"])
def test_bias_qkv_keeps_reference_bits(pattern):
    torch.manual_seed(5090)
    x = values((512, 3456), pattern)
    bias = torch.randn(3456, device="cuda", dtype=torch.bfloat16)
    if pattern == "zero": x.fill_(-0.0); bias.fill_(-0.0)
    reference_input = x.clone()
    zero = torch.zeros_like(x)
    reference = [torch.empty((512, 1152), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    actual = [torch.empty_like(v) for v in reference]
    old.bias_residual(reference_input.data_ptr(), zero.data_ptr(), bias.data_ptr(), 512, 3456)
    old.qkv_split(reference_input.data_ptr(), *[v.data_ptr() for v in reference], 512, 1152, 1152, 1152)
    fusion.bias_qkv(x.data_ptr(), bias.data_ptr(), *[v.data_ptr() for v in actual], 512, 1152)
    for left, right in zip(reference, actual): identical(left, right)


@pytest.mark.parametrize("pattern", ["0.01", "1", "16", "zero", "words"])
@pytest.mark.parametrize("vector", [4, 8])
@pytest.mark.parametrize("scale_value", [.0031, 1.0])
def test_bias_gelu_static_fp8_keeps_both_bf16_rounds(pattern, vector, scale_value):
    torch.manual_seed(5090)
    x = values((512, 4304), pattern)
    bias = torch.randn(4304, device="cuda", dtype=torch.bfloat16)
    if pattern == "zero": x.fill_(-0.0); bias.fill_(-0.0)
    scale = torch.tensor([scale_value], device="cuda", dtype=torch.float32)
    reference_input = x.clone()
    zero = torch.zeros_like(x)
    reference = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    actual = torch.empty_like(reference)
    old.bias_residual(reference_input.data_ptr(), zero.data_ptr(), bias.data_ptr(), 512, 4304)
    old.gelu_inplace(reference_input.data_ptr(), reference_input.numel())
    old.quantize_fp8_static(reference_input.data_ptr(), reference.data_ptr(), scale.data_ptr(), reference_input.numel())
    fusion.bias_gelu_fp8(x.data_ptr(), bias.data_ptr(), actual.data_ptr(), scale.data_ptr(), 512, 4304, vector)
    identical(reference, actual)


def test_fusion_guards_and_graph_capture():
    x = torch.zeros((512, 4304), device="cuda", dtype=torch.bfloat16)
    b = torch.zeros_like(x[0])
    y = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda")
    args = (x.data_ptr(), b.data_ptr(), y.data_ptr(), scale.data_ptr())
    assert not hasattr(fusion, "layernorm_fp8"), "unqualified reduction fusion must not be exposed"
    with pytest.raises(ValueError, match="shape"): fusion.bias_gelu_fp8(*args, 511, 4304)
    with pytest.raises(ValueError, match="overlapping"):
        fusion.bias_gelu_fp8(x.data_ptr(), b.data_ptr(), x.data_ptr(), scale.data_ptr(), 512, 4304)
    with pytest.raises(ValueError, match="misaligned"):
        fusion.bias_gelu_fp8(x.data_ptr() + 1, *args[1:], 512, 4304)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        fusion.bias_gelu_fp8(*args, 512, 4304, 8, stream.cuda_stream)
    graph.replay()
    torch.cuda.synchronize()
    assert not y.view(torch.uint8).any()
