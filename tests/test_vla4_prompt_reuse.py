"""CPU lifecycle checks for the real VLA4 prompt/capture method bodies.

Only CUDA allocation/capture and model kernels are replaced. These tests prove
storage/reset/calibration control flow; real action parity requires the GPU gate.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import logging
import time

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "serving/flash_rt/frontends/torch/vla4b_thor.py"


class Capture:
    def __init__(self, cuda):
        self.cuda, self.call = cuda, None

    def capture_begin(self):
        self.cuda.active = self

    def capture_end(self):
        self.cuda.active = None

    def replay(self):
        self.call()


class CpuTorch:
    def __init__(self):
        self.syncs = 0
        self.cuda = SimpleNamespace(active=None, synchronize=self.synchronize,
            Stream=lambda: SimpleNamespace(cuda_stream=17), stream=lambda _: nullcontext())
        self.cuda.CUDAGraph = lambda: Capture(self.cuda)

    def synchronize(self):
        self.syncs += 1

    def __getattr__(self, name):
        original = getattr(torch, name)
        if name in {"zeros", "empty", "full"}:
            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                return original(*args, **kwargs)
            return allocate
        return original


@pytest.fixture
def frontend_factory(monkeypatch):
    assert not torch.cuda.is_initialized()
    # Gathered CPU embeddings have fresh storage; clone models the upload's
    # ownership without invoking Tensor.cuda or importing a native extension.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self.clone())
    tree = ast.parse(SOURCE.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "Vla4bTorchFrontendThor")
    methods = {"set_prompt", "_reset_prompt_buffers", "_alloc_prompt_buffers",
               "_capture_graphs", "stage_inputs", "infer_staged"}
    definition.body = [n for n in definition.body if
        isinstance(n, ast.FunctionDef) and n.name in methods or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id.startswith("_PROMPT_") for t in n.targets)]
    cpu = CpuTorch()
    namespace = dict(torch=cpu, np=np, Union=Union, time=time,
        logger=logging.getLogger(__name__), fp16=torch.float16,
        VIS_SM=3, VIS_S=6, PATCH_FLAT=4, VIS_H_PAD=4,
        LM_L=2, LM_D=4, LM_H=8, LM_NH=2, KV_ROW=4,
        EXP_L=2, EXP_D=4, EXP_H=8, CHUNK=2, ADIM=3, SDIM=3, SUF=3)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
    cls = namespace["Vla4bTorchFrontendThor"]

    def make(graphs=True):
        fr = cls.__new__(cls)
        fr.use_cuda_graph, fr.graph_captured = graphs, False
        fr._prompt_token_count = None
        fr.prompt_updates = fr.prompt_buffer_reuses = fr.prompt_graph_captures = 0
        fr._real_data_calibrated = False
        fr.latency_records, fr.calibrations = [], []
        fr.builds = dict(rope=0, style=0, attention=0)
        fr._embed_cpu = torch.arange(160, dtype=torch.float16).reshape(40, 4) / 100
        fr._patches = torch.zeros(6, 4, dtype=torch.float16)
        fr._vis_emb = torch.zeros(3, 4, dtype=torch.float16)
        fr._patch_perm_cpu, fr._reverse_index = torch.arange(6), torch.arange(3)

        def tables(se):
            fr.builds["rope"] += 1
            for name, rows in (("_pre_cos", se), ("_pre_sin", se),
                               ("_suf_cos", 3), ("_suf_sin", 3)):
                setattr(fr, name, torch.arange(rows * 4).reshape(rows, 4))

        def styles():
            fr.builds["style"] += 1
            for name in ("_style_attn", "_style_ffn", "_t_contrib"):
                setattr(fr, name, torch.ones(2, 4))

        def attention():
            fr.builds["attention"] += 1
            fr._attn = SimpleNamespace(k=fr._Kc, v=fr._Vc)

        def vision(_):
            target, patches = fr._vis_emb, fr._patches
            run = lambda: target.fill_(patches.float().mean())
            if cpu.cuda.active is not None:
                cpu.cuda.active.call = run
            run()

        def lm_expert(_, calibrate=False):
            # Capture retained tensor objects, as native graphs retain their
            # addresses. Rebinding a Python attribute cannot update this graph.
            fields = ("_lang_emb", "_vis_emb", "_state_in", "_x_t",
                      "_lm_act_scales", "_exp_act_scales", "_Kc", "_Vc")
            captured = {name: getattr(fr, name) for name in fields}

            def run():
                t = captured
                signal = t["_lang_emb"].float().mean() + t["_vis_emb"].float().mean()
                t["_Kc"].fill_(signal)
                t["_Vc"].fill_(signal)
                if calibrate:
                    fr.calibrations.append({name: t[name].clone() for name in
                        ("_lang_emb", "_vis_emb", "_state_in", "_x_t")})
                    t["_lm_act_scales"].fill_(signal)
                    t["_exp_act_scales"].fill_(t["_x_t"].float().abs().mean() + .1)
                t["_x_t"].add_(signal + t["_state_in"].float().mean()
                    + t["_lm_act_scales"].mean() + t["_exp_act_scales"].mean())

            if cpu.cuda.active is not None:
                cpu.cuda.active.call = run
            run()

        fr._build_lm_tables, fr._build_style_tables = tables, styles
        fr._build_attn, fr._run_vis, fr._run_lm_expert = attention, vision, lm_expert
        return fr

    yield make, cpu
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("graphs", [False, True])
def test_same_length_keeps_all_tensor_and_graph_addresses_and_resets_state(frontend_factory, graphs):
    make, cpu = frontend_factory
    fr = make(graphs)
    fr.set_prompt([1, 2, 3])
    original = {name: value for name, value in vars(fr).items() if isinstance(value, torch.Tensor)}
    graphs_before = (getattr(fr, "_g_vis", None), getattr(fr, "_g_lm", None), fr._attn)
    for name in fr._PROMPT_ZERO_BUFFERS + fr._PROMPT_UNIT_BUFFERS:
        getattr(fr, name).fill_(7)
    fr._real_data_calibrated = True
    syncs = cpu.syncs
    fr.set_prompt([4, 5, 6])
    assert all(getattr(fr, name) is value for name, value in original.items())
    assert graphs_before == (getattr(fr, "_g_vis", None), getattr(fr, "_g_lm", None), fr._attn)
    assert torch.equal(fr._lang_emb, fr._embed_cpu[torch.tensor([4, 5, 6])])
    assert all(torch.count_nonzero(getattr(fr, name)) == 0 for name in fr._PROMPT_ZERO_BUFFERS)
    assert all(torch.equal(getattr(fr, name), torch.ones_like(getattr(fr, name)))
               for name in fr._PROMPT_UNIT_BUFFERS)
    assert not fr._real_data_calibrated and cpu.syncs == syncs + 1
    assert fr.builds == dict(rope=1, style=1, attention=1)
    assert (fr.prompt_updates, fr.prompt_buffer_reuses, fr.prompt_graph_captures) == (2, 1, int(graphs))


def test_reset_list_covers_every_initialized_prompt_buffer():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    allocation = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_alloc_prompt_buffers")
    initialized = {kind: set() for kind in ("zeros", "full")}
    for node in ast.walk(allocation):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr in initialized):
            initialized[node.value.func.attr].add(node.targets[0].attr)
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert set(constants["_PROMPT_ZERO_BUFFERS"]) == initialized["zeros"]
    assert set(constants["_PROMPT_UNIT_BUFFERS"]) == initialized["full"]


@pytest.mark.parametrize("graphs", [False, True])
def test_changed_length_retains_full_rebuild(frontend_factory, graphs):
    make, _ = frontend_factory
    fr = make(graphs)
    fr.set_prompt([1, 2, 3])
    old = {name: getattr(fr, name) for name in
        ("_lang_emb", "_Kc", "_pre_cos", "_style_attn", "_lm_act_scales", "_attn")}
    fr.set_prompt([4, 5, 6, 7])
    assert all(getattr(fr, name) is not value for name, value in old.items())
    assert (fr.Se, fr.total_keys) == (7, 10)
    assert fr.builds == dict(rope=2, style=2, attention=2)
    assert (fr.prompt_updates, fr.prompt_buffer_reuses, fr.prompt_graph_captures) == (2, 0, 2 * int(graphs))


@pytest.mark.parametrize("graphs", [False, True])
def test_updated_prompt_recalibrates_actual_inputs_and_restores_noise(frontend_factory, graphs):
    make, _ = frontend_factory
    optimized, fresh = make(graphs), make(graphs)
    patches, state, noise = torch.ones(6, 4), torch.ones(3), torch.arange(6).reshape(2, 3).float()
    optimized.set_prompt([1, 2, 3])
    optimized.infer_staged(patches, state, noise, vision_embeddings=torch.ones(3, 4))
    optimized.set_prompt([4, 5, 6])
    fresh.set_prompt([4, 5, 6])
    vision = torch.full((3, 4), 2.0)
    actual = optimized.infer_staged(patches * 3, state * 4, noise + 5, vision_embeddings=vision)
    expected = fresh.infer_staged(patches * 3, state * 4, noise + 5, vision_embeddings=vision)
    assert np.array_equal(actual["actions"], expected["actions"])
    assert len(optimized.calibrations) == 2 and len(fresh.calibrations) == 1
    for name, value in fresh.calibrations[-1].items():
        assert torch.equal(optimized.calibrations[-1][name], value)
    assert torch.equal(optimized.calibrations[-1]["_x_t"], (noise + 5).half())
    for name in optimized._PROMPT_UNIT_BUFFERS:
        assert torch.equal(getattr(optimized, name), getattr(fresh, name))
    optimized.infer_staged(patches, state, noise, vision_embeddings=vision)
    assert len(optimized.calibrations) == 2  # unchanged prompt keeps original calibration policy


def test_switching_graph_mode_requires_capture(frontend_factory):
    make, _ = frontend_factory
    fr = make(False)
    fr.set_prompt([1, 2, 3])
    fr.use_cuda_graph = True
    fr.set_prompt([4, 5, 6])
    assert fr.graph_captured and fr.prompt_graph_captures == 1
    assert fr.prompt_buffer_reuses == 0
