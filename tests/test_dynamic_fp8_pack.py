"""Packing optimization must preserve the declared E4M3 scale and bytes."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FP8 packing')
def test_dynamic_pack_matches_reference_including_graph_replay():
    if torch.cuda.get_device_capability() not in ((8, 9), (9, 0), (11, 0)):
        pytest.skip('requires SM89/SM90/SM110')
    pytest.importorskip('triton')
    from instinctflash.runtime.fp8_pack import dynamic_pack_bf16_e4m3
    def check(x):
        actual, scale = dynamic_pack_bf16_e4m3(x)
        expected_scale = x.abs().amax().float().clamp_min(1e-12).reshape(1) / 448.
        expected = (x.float() / expected_scale).clamp(-448,448).to(torch.float8_e4m3fn)
        assert torch.equal(scale, expected_scale)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for shape, magnitude in (((40,2048),1), ((256,5120),1), ((1,16),0),
                             ((1,16),1e-20), ((1,16),1e30)):
        check(torch.randn(shape,device='cuda',dtype=torch.bfloat16)*magnitude)
    # Every finite BF16 bit pattern, including subnormals and signed zero.
    patterns = torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    check(patterns[torch.isfinite(patterns)].reshape(1,-1).contiguous())
    x=torch.randn((40,2048),device='cuda',dtype=torch.bfloat16)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): dynamic_pack_bf16_e4m3(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):packed,scale=dynamic_pack_bf16_e4m3(x)
    for magnitude in (0,1e-4,20):
        x.copy_(torch.randn_like(x)*magnitude);graph.replay()
        eager,eager_scale=dynamic_pack_bf16_e4m3(x)
        assert torch.equal(scale,eager_scale)
        assert torch.equal(packed.view(torch.uint8),eager.view(torch.uint8))
    x.fill_(float('nan'))
    _,scale=dynamic_pack_bf16_e4m3(x)
    assert torch.isnan(scale).all()
