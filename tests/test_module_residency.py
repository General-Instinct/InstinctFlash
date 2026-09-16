"""Real CPU tensor checks for frozen storage ownership; GPU fit is separate."""
import pytest
import torch

from instinctflash.runtime.module_residency import ModuleResidency, resident_prefix


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.register_buffer("gain", torch.tensor(0.5, dtype=torch.bfloat16))
        self.register_buffer("scratch", torch.tensor(0.25), persistent=False)
        self.register_buffer("absent", None, persistent=False)
        self.failure = False
        self.mutate_buffer = False

    def forward(self, x, *, factor=1):
        if self.failure:
            raise ValueError("native block failed")
        if self.mutate_buffer:
            self.scratch.add_(1)
        return self.linear(x) * self.gain * factor + self.scratch


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = torch.nn.ModuleList([Block(), Block(), Block()])

    def forward(self, x):
        for block in self.tower:
            x = block(x, factor=2)
        return x


def install(net):
    # Three equal blocks, budget exactly one block: all blocks stream. CPU
    # copies exercise the ownership logic without claiming CUDA qualification.
    size = sum(t.numel() * t.element_size()
               for t in list(net.tower[0].parameters()) + list(net.tower[0].buffers()))
    return ModuleResidency(net, net.tower, device="cpu", budget_bytes=size,
                           reserve_bytes=0)


def test_budget_reserves_largest_streaming_block():
    assert resident_prefix([40, 20, 30], 10, 100) == 3
    assert resident_prefix([40, 20, 30], 10, 80) == 1
    assert resident_prefix([40, 20, 30], 10, 50) == 0
    with pytest.raises(RuntimeError, match="one streaming block"):
        resident_prefix([40, 20, 30], 10, 49)


def test_native_outputs_and_original_tensor_objects_survive_repeated_calls():
    net = Network().eval()
    inputs = torch.randn(4, 8)
    original = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    with torch.inference_mode():
        expected = net(inputs)
    owner = install(net)
    with torch.inference_mode():
        for _ in range(2):
            torch.testing.assert_close(net(inputs), expected, atol=0, rtol=0)
            assert all(value is original[name] for name, value in net.named_parameters())
            assert all(value is buffers[name] for name, value in net.named_buffers())
    assert owner.report()["streamed_block_calls"] == 6
    assert owner.report()["streamed_blocks"] == [0, 1, 2]
    assert net.tower[0].gain.dtype == torch.bfloat16
    assert net.tower[0].scratch.dtype == torch.float32
    assert net.tower[0].absent is None
    owner.close()
    owner.close()
    assert not any(layer._forward_hooks or layer._forward_pre_hooks for layer in net.tower)


def test_exception_restores_masters_and_allows_a_later_call():
    net = Network().eval()
    original = net.tower[1].linear.weight
    owner = install(net)
    net.tower[1].failure = True
    with torch.inference_mode(), pytest.raises(ValueError, match="native block failed"):
        net(torch.randn(1, 8))
    assert owner.active is None
    assert net.tower[1].linear.weight is original
    net.tower[1].failure = False
    with torch.inference_mode():
        assert torch.isfinite(net(torch.randn(1, 8))).all()
    owner.close()


def test_kwargs_and_output_hooks_keep_their_semantics_and_ownership():
    net = Network().eval()
    before = net.tower[0].register_forward_pre_hook(
        lambda module, args, kwargs: (args, {**kwargs, "factor": 3}), with_kwargs=True)
    after = net.tower[0].register_forward_hook(lambda module, args, output: output + 7)
    inputs = torch.randn(1, 8)
    with torch.inference_mode():
        expected = net(inputs)
    owner = install(net)
    with torch.inference_mode():
        torch.testing.assert_close(net(inputs), expected, atol=0, rtol=0)
    owner.close()
    assert len(net.tower[0]._forward_hooks) == 1
    assert len(net.tower[0]._forward_pre_hooks) == 1
    before.remove()
    after.remove()


def test_nonpersistent_buffer_mutation_refused_without_corrupting_cpu_master():
    net = Network().eval()
    original = net.tower[0].scratch
    value = original.clone()
    owner = install(net)
    net.tower[0].mutate_buffer = True
    with torch.inference_mode(), pytest.raises(RuntimeError, match="mutated a registered tensor"):
        net(torch.randn(1, 8))
    assert net.tower[0].scratch is original
    torch.testing.assert_close(original, value, atol=0, rtol=0)
    assert owner.active is None
    owner.close()


def test_training_and_grad_enabled_calls_are_refused_before_transfer():
    net = Network().eval()
    owner = install(net)
    with pytest.raises(RuntimeError, match="inference only"):
        net(torch.randn(1, 8))
    net.train()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference only"):
        net(torch.randn(1, 8))
    assert owner.calls == 0
    owner.close()


def test_non_descendant_duplicate_and_nested_blocks_are_refused():
    net = Network().eval()
    for blocks in ([Block()], [net.tower[0], net.tower[0]],
                   [net.tower[0], net.tower[0].linear], [net]):
        with pytest.raises(ValueError, match="nonoverlapping descendants"):
            ModuleResidency(net, blocks, device="cpu", budget_bytes=10000, reserve_bytes=0)


def test_cross_block_shared_storage_refused_before_placement():
    net = Network().eval()
    net.tower[1].linear.weight = net.tower[0].linear.weight
    with pytest.raises(ValueError, match="shared decoder-layer storage"):
        install(net)


def test_distinct_tensor_views_within_one_block_refused_before_placement():
    net = Network().eval()
    shared = torch.arange(8.)
    net.tower[0].register_buffer("first_view", shared[:4])
    net.tower[0].register_buffer("second_view", shared[4:])
    original = net.tower[0].linear.weight
    with pytest.raises(ValueError, match="distinct tensor views"):
        install(net)
    assert net.tower[0].linear.weight is original


def test_public_close_refuses_active_forward_and_leaves_owner_usable():
    net = Network().eval()
    owner = install(net)
    original = net.tower[0].linear.weight
    with torch.inference_mode():
        owner.enter(net.tower[0])
        borrowed = net.tower[0].linear.weight
        with pytest.raises(RuntimeError, match="active forward"):
            owner.close()
        assert net.tower[0].linear.weight is borrowed
        assert not owner.closed
        owner.leave(net.tower[0])
    assert net.tower[0].linear.weight is original
    owner.close()
