"""CUDA parity for native BF16 weight staging, independent of GEMM experiments."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("requires SM120", allow_module_level=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/dreamzero"))
from dreamzero_iwm.sm120_native import Stage


@pytest.mark.parametrize("rows", [1, 51, 880])
def test_native_bf16_streamed_weight_values_outputs_and_lifetime(rows):
    torch.manual_seed(120)
    resident = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(),
                                   torch.nn.Linear(512, 256)).bfloat16().eval().requires_grad_(False).cuda()
    import copy
    staged = copy.deepcopy(resident).cpu()
    masters = {k: v.clone() for k, v in staged.state_dict().items()}
    owner = SimpleNamespace(device=torch.device("cuda:0"), active=None, closed=False,
                            lock=threading.RLock(), synchronize=lambda: torch.cuda.synchronize(0))
    hook = Stage(owner, staged, "block")
    with torch.inference_mode():
        for _ in range(3):
            x = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
            expected, actual = resident(x), staged(x)
            torch.cuda.synchronize()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert all(p.device.type == "cpu" for p in staged.parameters())
            assert all(torch.equal(v, masters[k]) for k, v in staged.state_dict().items())
    assert hook.calls == 3 and owner.active is None
    hook.close()
