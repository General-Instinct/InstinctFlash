"""Explicit real-GPU tests; not part of the GitHub CPU gate."""
import pytest
import sys
from pathlib import Path

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("requires a real SM120 GPU", allow_module_level=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))

from flash_rt import flash_rt_kernels as kernels


@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_real_gemm_cache_restores_exact_output_and_refuses_new_shapes(precision):
    torch.manual_seed(120)
    m, n, k = 16, 32, 64
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k) if precision == "fp8" else (k, n), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    if precision == "fp8":
        a, b = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
    runner = kernels.GemmRunner()
    runner.set_cache_policy("record")

    def run(value):
        if precision == "fp8":
            value.fp8_nt_dev(a.data_ptr(), b.data_ptr(), output.data_ptr(), m, n, k, scale.data_ptr(), scale.data_ptr())
        else:
            value.bf16_nn(a.data_ptr(), b.data_ptr(), output.data_ptr(), m, n, k)
        torch.cuda.synchronize()
        return output.clone()

    expected = run(runner)
    records = runner.export_algo_cache()
    restored = kernels.GemmRunner()
    restored.import_algo_cache(runner.algo_cache_identity(), records)
    assert torch.equal(expected, run(restored))
    assert restored.export_algo_cache() == records
    with pytest.raises(RuntimeError, match="unregistered"):
        restored.bf16_nn(a.data_ptr(), b.data_ptr(), output.data_ptr(), m + 1, n, k)
    with pytest.raises(RuntimeError, match="cannot unlock"):
        restored.set_cache_policy("normal")


def test_restore_rejects_incompatible_corrupt_and_duplicate_records_transactionally():
    runner = kernels.GemmRunner()
    with pytest.raises(ValueError, match="environment"):
        runner.import_algo_cache("different", [(0, 16, 16, 16, bytes(64))])
    with pytest.raises(ValueError, match="malformed"):
        runner.import_algo_cache(runner.algo_cache_identity(), [(99, 16, 16, 16, bytes(64))])
    assert not runner.export_algo_cache()
    source = kernels.GemmRunner()
    a = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)
    source.bf16_nn(a.data_ptr(), a.data_ptr(), out.data_ptr(), 16, 16, 16)
    torch.cuda.synchronize()
    rows = source.export_algo_cache()
    with pytest.raises(ValueError, match="duplicate"):
        runner.import_algo_cache(source.algo_cache_identity(), rows * 2)
    assert not runner.export_algo_cache()
    runner.import_algo_cache(source.algo_cache_identity(), rows)
    assert runner.export_algo_cache() == rows


def test_record_policy_does_not_retune_a_shape_after_first_capture():
    runner = kernels.GemmRunner()
    runner.set_cache_policy("record")
    a = torch.ones((16, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((64, 32), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((16, 32), device="cuda", dtype=torch.bfloat16)
    runner.autotune_bf16_nn(a.data_ptr(), b.data_ptr(), output.data_ptr(), 16, 32, 64, 2)
    recorded = runner.export_algo_cache()
    # A zero-candidate request would be invalid if the tuner were invoked again.
    runner.autotune_bf16_nn(a.data_ptr(), b.data_ptr(), output.data_ptr(), 16, 32, 64, 0)
    assert runner.export_algo_cache() == recorded
    runner.set_cache_policy("frozen")
    runner.autotune_bf16_nn(a.data_ptr(), b.data_ptr(), output.data_ptr(), 16, 32, 64, 0)
    assert runner.export_algo_cache() == recorded
