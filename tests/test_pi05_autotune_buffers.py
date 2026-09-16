"""Exercise real autotune/selection methods with capacity-checked CPU buffer doubles.

Extracting just these methods avoids importing CUDA libraries on the CPU gate.
The test checks the actual GEMM M*K reads, not a copy of the selection rule.
"""
import ast
from collections import defaultdict
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def pipeline_classes():
    namespace = {"logger": logging.getLogger(__name__)}
    classes = []
    for filename, name, methods in (
        ("pipeline_rtx.py", "Pi05Pipeline", {"_pick_fp8_scratch", "_weight_fp8", "autotune_gemms"}),
        ("pipeline_rtx_batched.py", "Pi05BatchedPipeline", {"_pick_fp8_scratch_b2", "autotune_gemms"}),
    ):
        tree = ast.parse((ROOT / "serving/flash_rt/models/pi05" / filename).read_text())
        constants = [n for n in tree.body if isinstance(n, ast.Assign) and
                     isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith(("VIS_", "ENC_", "DEC_"))]
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        module = ast.fix_missing_locations(ast.Module(body=constants + [cls], type_ignores=[]))
        exec(compile(module, filename, "exec"), namespace)
        classes.append(namespace[name])
    return classes


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("chunk", [10, 50])
def test_every_autotuned_gemm_has_a_full_m_by_k_activation_buffer(batch, chunk):
    classes = pipeline_classes()
    p = classes[batch - 1]()
    p.B = 2
    p.num_views, p.vision_seq, p.encoder_seq_len, p.chunk_size = 2, 512, 665, chunk
    p.use_fp8 = p.use_fp8_decoder = p.fp8_calibrated = True
    buffers = {}

    def buffer(nbytes=10**9):
        pointer = len(buffers) + 1
        result = SimpleNamespace(ptr=SimpleNamespace(value=pointer), nbytes=nbytes)
        buffers[pointer] = result
        return result

    p.bufs = defaultdict(buffer)
    for suffix, multiplier in (("", 1), ("_b2", 2)):
        for domain, rows, width, large in (("vis", 512, 1152, 4304),
                                          ("enc", 665, 2048, 2 * 16384),
                                          ("dec", chunk, 1024, 2 * 4096)):
            p.bufs[f"{domain}_act_fp8{suffix}"] = buffer(multiplier * rows * width)
            p.bufs[f"{domain}_act_fp8_large{suffix}"] = buffer(multiplier * rows * large)
    names = [f"{family}_{kind}_w_0" for family in ("vision", "encoder", "decoder")
             for kind in ("attn_qkv", "attn_o", "ffn_up" if family == "vision" else "ffn_gate_up", "ffn_down")]
    names.append("vision_projector_w")
    p.weights = {"fp8": {name: (1, 1) for name in names}, "vision_patch_embedding_w": 1,
                 "encoder_ffn_down_w": [0] * 18}
    p.fp8_act_scales = {name: buffer(4) for name in names}
    p.gemm = SimpleNamespace(autotune_bf16_nn=lambda *args: None)
    p._cudart = SimpleNamespace(cudaDeviceSynchronize=lambda: None)
    observed = []

    def check_read(pointer, weight, output, m, n, k, *scales):
        assert buffers[pointer].nbytes >= m * k, (m, n, k, buffers[pointer].nbytes)
        observed.append((m, n, k, pointer))

    p._autotune_fp8_matmul = check_read
    p.autotune_gemms()
    suffix = "_b2" if batch == 2 else ""
    attention = next(row for row in observed if row[:3] == (batch * chunk, 1024, 2048))
    assert attention[3] == p.bufs[f"dec_act_fp8_large{suffix}"].ptr.value
    pick = p._pick_fp8_scratch_b2 if batch == 2 else p._pick_fp8_scratch
    with pytest.raises(ValueError, match="too small"):
        pick("decoder_attn_o_w_0", p.bufs[f"dec_act_fp8_large{suffix}"].nbytes + 1)
