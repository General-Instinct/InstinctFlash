from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from instinctflash.runtime.capture_self_check import compare_tensors, run_capture_self_check
from run_tests import run_module_tests


def test_strict_bytes_reject_roundoff_signed_zero_and_dtype_changes():
    for a,b in [(torch.tensor([0.]),torch.tensor([-0.])),
                (torch.tensor([1.],dtype=torch.float64),torch.tensor([1.+2**-40],dtype=torch.float64)),
                (torch.tensor([1.]),torch.tensor([1.],dtype=torch.float64))]:
        assert not compare_tensors(a,b)['bitexact']
    a=torch.arange(12.).reshape(3,4).T
    assert compare_tensors(a,a.contiguous())['bitexact']


def test_no_evidence_or_nonfinite_is_never_a_pass():
    assert not run_capture_self_check(family='test',cases=[])['passed']
    for value in [float('nan'),float('inf'),-float('inf')]:
        a=torch.tensor([value]); b=torch.tensor([0.])
        assert not run_capture_self_check(family='test',cases=[('invalid',lambda:a,lambda:b)])['passed']
    assert not compare_tensors(torch.empty(0),torch.empty(0))['valid']
    assert not compare_tensors(torch.ones(2),torch.ones(1,2))['valid']

if __name__=='__main__': raise SystemExit(run_module_tests(globals()))
