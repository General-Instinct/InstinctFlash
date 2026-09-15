"""CPU checks of VLA2's actual prompt, calibration and graph lifecycle methods.

CUDA allocation/capture and model kernels are replaced with CPU fixtures. This
checks storage and control flow, not the numerical quality of real model actions.
"""
import ast
from contextlib import contextmanager, nullcontext
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import numpy as np
import pytest
import torch

from instinctflash.runtime.vla2_engine import Vla2StagedEngine


SOURCE = Path(__file__).resolve().parents[1] / "serving/flash_rt/frontends/torch/vla2_thor.py"


class Capture:
    def __init__(self, cuda):
        self.cuda, self.calls = cuda, []

    def capture_begin(self):
        assert self.cuda.active is None
        self.cuda.active = self

    def capture_end(self):
        self.cuda.active = None

    def replay(self):
        for call in self.calls:
            call()


class CpuTorch:
    def __init__(self):
        self.syncs = 0
        self.cuda = SimpleNamespace(active=None, synchronize=self.synchronize,
            Stream=lambda **_: SimpleNamespace(cuda_stream=17),
            default_stream=lambda _: SimpleNamespace(cuda_stream=0),
            stream=lambda _: nullcontext(), graph=self.graph)
        self.cuda.CUDAGraph = lambda: Capture(self.cuda)

    @contextmanager
    def graph(self, graph, **_):
        graph.capture_begin()
        try:
            yield
        finally:
            graph.capture_end()

    def synchronize(self):
        self.syncs += 1

    def run(self, call):
        if self.cuda.active is not None:
            self.cuda.active.calls.append(call)
        call()

    def __getattr__(self, name):
        original = getattr(torch, name)
        if name in {"zeros", "empty", "full"}:
            def allocate(*args, **kwargs):
                kwargs["device"] = "cpu"
                return original(*args, **kwargs)
            return allocate
        return original


@pytest.fixture
def engine_factory(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self.clone())
    cpu = CpuTorch()
    for name in ("synchronize", "Stream", "default_stream", "stream", "graph", "CUDAGraph"):
        monkeypatch.setattr(torch.cuda, name, getattr(cpu.cuda, name))
    tree = ast.parse(SOURCE.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "Vla2TorchFrontendThor")
    methods = {"set_prompt", "_reset_prompt_buffers", "_alloc_prompt_buffers",
               "_capture_graphs", "stage_inputs"}
    definition.body = [n for n in definition.body if
        isinstance(n, ast.FunctionDef) and n.name in methods or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id.startswith("_PROMPT_") for t in n.targets)]
    namespace = dict(torch=cpu, np=np, Union=Union, logger=logging.getLogger(__name__),
        fp16=torch.float16, IMG_BLOCK=2, ALIGN_TOKENS=1,
        LM_L=2, LM_D=4, LM_H=8, LM_DQ=8, LM_DKV=4, LM_QKV_OUT=16,
        LM_NH=2, KV_ROW=4, EXP_L=2, EXP_D=4, SHARED_H=4,
        CHUNK=2, ADIM=3, SDIM=3, SUF=3, VIS_S=6, VIS_H=8, PATCH_FLAT=4)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SOURCE), "exec"), namespace)
    cls = namespace["Vla2TorchFrontendThor"]

    def make(graphs=True, frontend_graphs=False):
        fr = cls.__new__(cls)
        fr.num_views, fr.use_cuda_graph, fr.graph_captured = 3, frontend_graphs, False
        fr._prompt_layout, fr._prompt_buffers_reused = None, False
        fr.prompt_updates = fr.prompt_buffer_reuses = fr.prompt_graph_captures = 0
        fr._real_data_calibrated, fr.lm_prefill_precision = False, "fp16"
        fr.latency_records, fr.calibration_inputs, fr.scale_traces = [], [], []
        fr.builds = dict(rope=0, style=0, attention=0)
        fr._embed_cpu = torch.zeros(151654, 4, dtype=torch.float16)
        fr._embed_cpu[:40] = torch.arange(160).reshape(40, 4) / 100
        # Only the two high-ID boundary rows need values outside the tiny vocabulary.
        fr._embed_cpu[151652:] = .5
        fr._patches = torch.zeros(6, 4, dtype=torch.float16)
        fr._vis_emb = torch.zeros(3, 4, dtype=torch.float16)
        fr._ds_out = [torch.zeros(3, 4, dtype=torch.float16) for _ in range(3)]
        moe = SimpleNamespace(dn_act=torch.ones(2), calibrate=False, router=[], slots=[])
        moe.routed_moe_fn = lambda *_, **__: None
        moe.bind_router_fp16 = lambda ptr: moe.router.append(ptr)
        moe.bind_ffn_slots = lambda ptr, stride: moe.slots.append((ptr, stride))

        def tables(n_lang):
            fr.builds["rope"] += 1
            for name, rows in (("_pre_cos", fr.Se), ("_pre_sin", fr.Se),
                               ("_suf_cos", 3), ("_suf_sin", 3)):
                setattr(fr, name, torch.full((rows, 4), n_lang))

        def styles():
            fr.builds["style"] += 1
            for name in ("_style_attn", "_style_ffn", "_t_contrib"):
                setattr(fr, name, torch.ones(2, 4))

        def attention():
            fr.builds["attention"] += 1
            fr._attn = SimpleNamespace(k=fr._Kc, v=fr._Vc, router=fr._s_xn)

        def prefill(_):
            # Retain objects as captured CUDA kernels retain raw pointers.
            lang, vis, deep = fr._lang_emb, fr._vis_emb, tuple(fr._ds_out)
            k, v = fr._Kc, fr._Vc

            def run():
                signal = lang.float().mean() + vis.float().mean()
                signal += sum(d.float().mean() for d in deep)
                k.fill_(signal)
                v.fill_(signal / 2)

            cpu.run(run)

        def expert(_, calibrate=False, calibration_observer=None):
            x, state, k = fr._x_t, fr._state_in, fr._Kc
            scales, down, router = fr._exp_act_scales, moe.dn_act, fr._s_xn

            def run():
                assert moe.router[-1] == router.data_ptr()
                assert moe.slots[-1] == (scales.data_ptr() + 8, 16)
                signal = k.float().mean() + state.float().mean()
                if calibrate:
                    assert moe.calibrate
                    fr.calibration_inputs.append(dict(
                        lang=fr._lang_emb.clone(), vision=fr._vis_emb.clone(),
                        deep=[v.clone() for v in fr._ds_out], state=state.clone(), noise=x.clone()))
                    # Early maxima must survive later lower values. Every
                    # scale slot and layer receives all ten observations.
                    history = []
                    base = signal.abs() / 100 + x.float().abs().mean() / 100 + .01
                    for step in range(10):
                        for layer in range(2):
                            values = base + torch.tensor([.01, .02, .03, .04]) * (10-step) * (layer+1)
                            scales.reshape(2, 4)[layer].copy_(values)
                            down[layer] = base + .02 * (10-step) * (layer+1)
                            calibration_observer(layer, step)
                        history.append((scales.clone(), down.clone()))
                        x.add_(.1)
                    fr.scale_traces.append(history)
                else:
                    assert not moe.calibrate
                    router.fill_(signal)
                    x.add_(signal / 10 + scales.mean() + down.mean())

            cpu.run(run)

        vision_calls = []

        def vision(patches):
            vision_calls.append(patches.clone())
            merged = torch.full((3, 4), patches.float().mean())
            return merged, [merged + i + 1 for i in range(3)]

        fr._build_lm_tables, fr._build_style_tables, fr._build_attn = tables, styles, attention
        fr._run_lm_prefill_only, fr._run_expert_only = prefill, expert
        # The frontend's optional two-graph route is separate from the staged
        # engine. Its fixture kernels test graph/storage lifecycle only.
        fr._run_vis = lambda _: cpu.run(lambda: fr._vis_emb.fill_(1))
        fr._run_lm_expert = lambda _: prefill(0)
        if frontend_graphs:
            return fr
        engine = Vla2StagedEngine(fr, moe, vision, use_cuda_graph=graphs)
        engine.vision_calls = vision_calls
        return engine

    yield make, cpu
    assert not torch.cuda.is_initialized()


def inputs(offset=0):
    return (torch.full((6, 4), 1. + offset), torch.full((3,), 2. + offset),
            torch.arange(6).reshape(2, 3).float() + offset)


@pytest.mark.parametrize("graphs", [False, True])
def test_same_length_retains_all_storage_and_rebinds_moe_after_exact_resets(engine_factory, graphs):
    make, cpu = engine_factory
    engine = make(graphs)
    engine.set_prompt([1, 2, 3])
    engine.infer_staged(*inputs())
    fr = engine.frontend
    original = {name: value for name, value in vars(fr).items() if isinstance(value, torch.Tensor)}
    old_graph, old_attn, down = engine._graph, fr._attn, engine.moe.dn_act.clone()
    for name in fr._PROMPT_ZERO_BUFFERS + fr._PROMPT_UNIT_BUFFERS:
        getattr(fr, name).fill_(7)
    fr._real_data_calibrated = True
    syncs = cpu.syncs
    rng = torch.random.get_rng_state().clone()
    engine.set_prompt([4, 5, 6])
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(getattr(fr, name) is value for name, value in original.items())
    assert fr._attn is old_attn and engine._graph is old_graph
    assert torch.equal(fr._lang_emb, fr._embed_cpu[torch.tensor([4, 5, 6])])
    assert all(torch.count_nonzero(getattr(fr, name)) == 0 for name in fr._PROMPT_ZERO_BUFFERS)
    assert all(torch.equal(getattr(fr, name), torch.ones_like(getattr(fr, name)))
               for name in fr._PROMPT_UNIT_BUFFERS)
    assert torch.equal(engine.moe.dn_act, down)  # Original owner is overwritten during calibration.
    assert not engine._calibrated and not fr._real_data_calibrated
    assert cpu.syncs == syncs + 1
    assert engine.moe.router == [fr._s_xn.data_ptr()] * 2
    assert engine.moe.slots == [(fr._exp_act_scales.data_ptr() + 8, 16)] * 2
    assert fr.builds == dict(rope=1, style=1, attention=1)
    assert (fr.prompt_updates, fr.prompt_buffer_reuses) == (2, 1)
    assert (engine.graph_captures, engine.prompt_graph_reuses) == (int(graphs), int(graphs))


def test_reset_list_covers_every_initialized_buffer_and_preserves_v2_scratch_capacity():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Vla2TorchFrontendThor")
    allocation = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_alloc_prompt_buffers")
    initialized = {kind: set() for kind in ("zeros", "full")}
    for node in ast.walk(allocation):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr in initialized):
            initialized[node.value.func.attr].add(node.targets[0].attr)
            if node.targets[0].attr == "_s_fp8":
                assert ast.unparse(node.value.args[0]) == "SUF * LM_DQ"
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert set(constants["_PROMPT_ZERO_BUFFERS"]) == initialized["zeros"]
    assert set(constants["_PROMPT_UNIT_BUFFERS"]) == initialized["full"]


@pytest.mark.parametrize("graphs", [False, True])
def test_changed_length_rebuilds_and_recaptures_with_new_moe_pointers(engine_factory, graphs):
    make, _ = engine_factory
    engine = make(graphs)
    engine.set_prompt([1, 2, 3])
    engine.infer_staged(*inputs())
    fr, old_graph = engine.frontend, engine._graph
    old = {name: getattr(fr, name) for name in
        ("_lang_emb", "_bound_emb", "_Kc", "_pre_cos", "_style_attn", "_exp_act_scales", "_s_xn", "_attn")}
    engine.set_prompt([4, 5, 6, 7])
    assert engine._graph is None and not fr.graph_captured
    assert all(getattr(fr, name) is not value for name, value in old.items())
    assert (fr.Se, fr.n_lang, fr.total_keys) == (11, 4, 14)
    assert engine.moe.router[-1] == fr._s_xn.data_ptr() != engine.moe.router[-2]
    assert engine.moe.slots[-1][0] == fr._exp_act_scales.data_ptr() + 8 != engine.moe.slots[-2][0]
    engine.infer_staged(*inputs(1))
    if graphs:
        assert engine._graph is not old_graph
    else:
        assert engine._graph is None
    assert engine.graph_captures == 2 * int(graphs)
    assert fr.builds == dict(rope=2, style=2, attention=2)
    assert (fr.prompt_updates, fr.prompt_buffer_reuses) == (2, 0)


@pytest.mark.parametrize("graphs", [False, True])
def test_new_prompt_recalibrates_current_observation_all_steps_and_restores_noise(engine_factory, graphs):
    make, _ = engine_factory
    reused, fresh = make(graphs), make(graphs)
    reused.set_prompt([1, 2, 3])
    previous = reused.infer_staged(*inputs())
    saved = previous["actions"].copy()
    reused.set_prompt([4, 5, 6])
    fresh.set_prompt([4, 5, 6])
    current = inputs(3)
    rng = torch.random.get_rng_state().clone()
    actual = reused.infer_staged(*current)
    expected = fresh.infer_staged(*current)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert np.array_equal(actual["actions"], expected["actions"])
    assert np.array_equal(previous["actions"], saved)  # Output owns its storage.
    fr = reused.frontend
    left, right = fr.calibration_inputs[-1], fresh.frontend.calibration_inputs[-1]
    for key in ("lang", "vision", "state", "noise"):
        assert torch.equal(left[key], right[key])
    assert all(torch.equal(a, b) for a, b in zip(left["deep"], right["deep"]))
    assert torch.equal(left["noise"], current[2].half())
    history = fr.scale_traces[-1]
    assert len(history) == 10
    assert torch.equal(fr._exp_act_scales, torch.stack([entry[0] for entry in history]).amax(0))
    assert torch.equal(reused.moe.dn_act, torch.stack([entry[1] for entry in history]).amax(0))
    assert torch.equal(fr._exp_act_scales, fresh.frontend._exp_act_scales)
    assert torch.equal(reused.moe.dn_act, fresh.moe.dn_act)
    assert reused.calibrations == 2 and fresh.calibrations == 1
    assert reused.graph_captures == int(graphs) and not reused.moe.calibrate
    assert len(reused.vision_calls) == 2
    # Same tokens retain the old calibration policy, but vision/state/noise
    # must still refresh on each observation and no prompt buffers are reset.
    reused.set_prompt([4, 5, 6])
    latest = reused.infer_staged(*inputs(6))
    assert reused.calibrations == 2 and len(reused.vision_calls) == 3
    assert fr.prompt_updates == 2 and fr.prompt_buffer_reuses == 1
    assert not np.array_equal(latest["actions"], actual["actions"])
    assert reused.replays == 3 * int(graphs)


def test_frontend_owned_graphs_reuse_and_graph_mode_change_rebuilds(engine_factory):
    make, _ = engine_factory
    fr = make(frontend_graphs=True)
    fr.set_prompt([1, 2, 3])
    original = (fr._g_vis, fr._g_lm, fr._attn)
    fr.set_prompt([4, 5, 6])
    assert (fr._g_vis, fr._g_lm, fr._attn) == original
    assert fr.prompt_graph_captures == 1 and fr.graph_captured
    fr.use_cuda_graph = False
    fr.set_prompt([7, 8, 9])
    assert not fr.graph_captured and not fr._prompt_buffers_reused
    assert fr._attn is not original[2]
    fr.use_cuda_graph = True
    fr.set_prompt([1, 2, 3])
    assert fr.prompt_graph_captures == 2 and fr.graph_captured


@pytest.mark.parametrize("failure", ["reset", "router", "ffn"])
def test_partial_prompt_failure_invalidates_graph_and_can_recover(engine_factory, monkeypatch, failure):
    make, _ = engine_factory
    engine = make()
    engine.set_prompt([1, 2, 3])
    engine.infer_staged(*inputs())
    fr = engine.frontend
    old_graph = engine._graph
    target, method = {
        "reset": (fr, "_reset_prompt_buffers"),
        "router": (engine.moe, "bind_router_fp16"),
        "ffn": (engine.moe, "bind_ffn_slots"),
    }[failure]
    original = getattr(target, method)

    def fail(*args):
        original(*args)
        raise RuntimeError("injected prompt failure")

    monkeypatch.setattr(target, method, fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine.set_prompt([4, 5, 6])
    assert engine._graph is None and engine._tokens is None
    if failure == "reset":
        assert fr._prompt_layout is None and not fr._prompt_buffers_reused
    assert not fr.graph_captured and not engine._calibrated
    with pytest.raises(RuntimeError, match="set_prompt"):
        engine.infer_staged(*inputs())
    monkeypatch.setattr(target, method, original)
    engine.set_prompt([4, 5, 6])
    engine.infer_staged(*inputs())
    assert engine._graph is not old_graph and engine.graph_captures == 2


def test_calibration_failure_never_replays_and_retries_current_observation(engine_factory, monkeypatch):
    make, _ = engine_factory
    engine = make()
    engine.set_prompt([1, 2, 3])
    engine.infer_staged(*inputs())
    engine.set_prompt([4, 5, 6])
    original = engine.frontend._run_expert_only

    def incomplete(stream, calibrate=False, calibration_observer=None):
        if calibrate:
            calibration_observer(0, 0)
        else:
            original(stream)

    monkeypatch.setattr(engine.frontend, "_run_expert_only", incomplete)
    with pytest.raises(RuntimeError, match="incomplete expert calibration"):
        engine.infer_staged(*inputs(3))
    assert not engine._calibrated and not engine.moe.calibrate
    assert engine.replays == 1 and engine.calibrations == 1
    monkeypatch.setattr(engine.frontend, "_run_expert_only", original)
    engine.infer_staged(*inputs(5))
    assert engine.replays == 2 and engine.calibrations == 2
    assert torch.equal(engine.frontend.calibration_inputs[-1]["noise"], inputs(5)[2].half())


def test_public_graph_statistics_expose_real_capture_reuse_and_calibration_counts(engine_factory):
    make, _ = engine_factory
    engine = make()
    for ids in ([1, 2, 3], [4, 5, 6]):
        engine.set_prompt(ids)
        engine.infer_staged(*inputs())
    engine.vision.capture = SimpleNamespace(graph=object(), verdict={"passed": True})
    path = SOURCE.parents[4] / "instinctflash/runtime/vla2_engine.py"
    tree = ast.parse(path.read_text())
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "EngineLoop")
    loop.bases = []
    loop.body = [n for n in loop.body if isinstance(n, ast.FunctionDef) and n.name == "graph_stats"]
    namespace = {}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), namespace)
    instance = namespace["EngineLoop"]()
    instance._server = SimpleNamespace(vla=SimpleNamespace(model=SimpleNamespace(engine=engine)))
    stats = instance.graph_stats
    assert stats["captured"] and stats["vision_graph"] and stats["language_action_graph"]
    assert stats["replays"] == 2 and stats["language_action_captures"] == 1
    assert stats["language_action_prompt_reuses"] == stats["prompt_buffer_reuses"] == 1
    assert stats["expert_calibrations"] == stats["prompt_updates"] == 2
