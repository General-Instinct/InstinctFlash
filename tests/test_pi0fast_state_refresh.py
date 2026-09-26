"""CPU contracts for the real Pi0-FAST wrapper and frontend methods.

Full frontend modules are loaded with native imports replaced. Their constructors
are bypassed; tensor storage, tokenizer, calibration, and graph capture are CPU
stand-ins. Real set_prompt/infer run up to prefill replay, where a snapshot replaces
model execution. These tests do not establish CUDA correctness or model quality.
"""
from collections import Counter
import ctypes
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVING = ROOT / "serving"
sys.path.insert(0, str(SERVING))
from flash_rt.api import VLAModel


class Tensor:
    """Small CPU storage stand-in; slices share storage, copy_ preserves it."""

    def __init__(self, values):
        self.values = np.asarray(values)
        self.copies = 0

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def __mul__(self, value):
        return Tensor(self.values * value)

    def long(self):
        return Tensor(self.values.astype(np.int64))

    def cuda(self):
        return self

    def copy_(self, other):
        assert self.shape == other.shape
        np.copyto(self.values, other.values)
        self.copies += 1
        return self

    def data_ptr(self):
        return self.values.__array_interface__["data"][0]


class Tokenizer:
    """Byte tokens make padding/length cases explicit, not SentencePiece claims."""

    def Encode(self, text):
        return list(text.encode("ascii"))

    def bos_id(self):
        return 1


class AtPrefill(Exception):
    def __init__(self, snapshot):
        self.snapshot = snapshot


def unexpected_native_call(*args, **kwargs):
    raise AssertionError("Native model execution is outside this CPU test")


@pytest.fixture(params=["torch", "jax"])
def frontend_factory(request, monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def embedding(ids, weights):
        weights.lookups += 1
        return Tensor(weights.values[ids.values])

    functional = stub("torch.nn.functional", embedding=embedding)
    nn = stub("torch.nn", functional=functional)
    stub("torch", nn=nn, Tensor=Tensor, float16=np.float16, float8_e4m3fn=object(),
         from_numpy=Tensor, cat=lambda tensors, dim=0:
         Tensor(np.concatenate([t.values for t in tensors], axis=dim)),
         cuda=SimpleNamespace(synchronize=lambda: None))
    stub("flash_rt.hardware.thor.shared_primitives",
         siglip_forward=unexpected_native_call, postln_project=unexpected_native_call)
    stub("flash_rt.models.pi0fast.pipeline", **{name: unexpected_native_call for name in (
        "prefill_forward_pi0fast", "decode_step_pi0fast", "decode_step_pi0fast_bf16",
        "prefill_calibrate_pi0fast", "siglip_forward_sm120")})
    stub("flash_rt.flash_rt_kernels")
    stub("flash_rt.core.cuda_buffer", CudaBuffer=object)
    stub("flash_rt.core.quant.calibrator", load_calibration=unexpected_native_call,
         save_calibration=unexpected_native_call)
    stub("flash_rt.core.thor_frontend_utils", quant_fp8=unexpected_native_call,
         interleave_qk=unexpected_native_call)

    framework = request.param
    path = SERVING / "flash_rt" / "frontends" / framework / "pi0fast.py"
    spec = importlib.util.spec_from_file_location(f"_state_test_{framework}", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    with monkeypatch.context() as loading:
        loading.setattr(ctypes, "CDLL", lambda name: SimpleNamespace())
        spec.loader.exec_module(module)
    cls = getattr(module, "Pi0FastTorchFrontend" if framework == "torch"
                  else "Pi0FastJaxFrontend")

    def build(decode_graph=True):
        pipe = object.__new__(cls)
        pipe.sig_S, pipe.num_views, pipe.De = 2, 1, 4
        pipe.He, pipe.Le, pipe.NHe, pipe.HDe = 8, 1, 1, 4
        pipe.Se_max, pipe.max_decode_steps = 256, 8
        pipe.decode_cuda_graph = decode_graph
        pipe.graph_captured = pipe.calibrated = pipe._real_data_calibrated = False
        pipe._sp_tokenizer = Tokenizer()
        pipe.embedding_weight = Tensor(np.repeat(np.arange(256)[:, None], 4, axis=1))
        pipe.embedding_weight.lookups = 0
        pipe._full_rope = Tensor(np.zeros((pipe.Se_max, 2)))
        pipe._enc_rope = Tensor(np.zeros((pipe.Se_max, 2)))
        calls = Counter()
        image = np.full((2, 2, 3), 0.5, dtype=np.float16)
        image_buffer = SimpleNamespace(current=None)

        def upload(values):
            image_buffer.current = np.array(values, copy=True)

        image_buffer.upload = upload
        pipe._img_buf = image_buffer

        def capture_vision():
            calls["vision_capture"] += 1
            # Like real capture warmup, this replaces the image buffer.
            upload(np.zeros((1, *image.shape), dtype=np.float16))
            bound = pipe._lang_emb
            captured_length = pipe.Se

            def replay():
                pipe._observed = SimpleNamespace(
                    embeddings=bound.values.copy(), pointer=bound.data_ptr(),
                    image=image_buffer.current.copy(), vision_length=captured_length)

            pipe._siglip_graph = SimpleNamespace(replay=replay)

        def calibrate(length, force=False):
            calls["calibration"] += 1
            calls["real_calibration"] += bool(force)
            pipe._enc_calib_scales = Tensor(np.ones(4) * calls["calibration"])

        def capture_prefill():
            calls["prefill_capture"] += 1
            captured_length = pipe.Se
            captured_scales = pipe._enc_calib_scales

            def replay():
                snapshot = pipe._observed
                snapshot.prefill_length = captured_length
                snapshot.scales_pointer = captured_scales.data_ptr()
                snapshot.ready = pipe._real_data_calibrated
                raise AtPrefill(snapshot)

            pipe._prefill_graph = SimpleNamespace(replay=replay)

        def capture_decode():
            calls["decode_capture"] += 1
            pipe._decode_graph = SimpleNamespace(prefill_length=pipe.Se)

        pipe._capture_siglip_graph = capture_vision
        pipe._calibrate = calibrate
        pipe._capture_prefill_graph = capture_prefill
        pipe._capture_decode_action_graph = capture_decode
        real_infer = pipe.infer

        def infer_to_boundary(observation, **kwargs):
            try:
                real_infer(observation, **kwargs)
            except AtPrefill as boundary:
                return {"actions": boundary.snapshot}
            raise AssertionError("Expected the real infer path to reach prefill replay")

        # Keep real infer execution, substituting only the GPU result. This also
        # lets the real single-observation calibration bootstrap finish normally.
        pipe.infer = infer_to_boundary
        model = VLAModel(pipe, framework)
        return SimpleNamespace(pipe=pipe, model=model, calls=calls, image=image)

    return build


def predict(harness, state=None, prompt=None):
    return harness.model.predict([harness.image], prompt=prompt, state=state)


def graph_objects(pipe):
    return (pipe._siglip_graph, pipe._prefill_graph, getattr(pipe, "_decode_graph", None))


def assert_current(snapshot, harness, reference):
    np.testing.assert_array_equal(snapshot.embeddings, reference.embeddings)
    np.testing.assert_array_equal(snapshot.image, [harness.image])
    assert snapshot.vision_length == snapshot.prefill_length == harness.pipe.Se
    assert snapshot.ready


@pytest.mark.parametrize("delivery", ["same_task", "omitted_task", "mapping", "in_place",
                                       "mapping_precedence"])
def test_current_state_reaches_captured_prefix(frontend_factory, delivery):
    h, fresh = frontend_factory(), frontend_factory()
    state = np.array([0.0], dtype=np.float32)
    first = predict(h, state, "pick up block")
    before, graphs = h.calls.copy(), graph_objects(h.pipe)
    if delivery == "in_place":
        state[:] = 0.75
    else:
        state = np.array([0.75], dtype=np.float32)
    if delivery.startswith("mapping"):
        obs = {"images": [h.image], "state": state}
        explicit = np.array([-0.75]) if delivery == "mapping_precedence" else None
        actual = h.model.predict(obs, state=explicit)
    else:
        actual = predict(h, state, "pick up block" if delivery == "same_task" else None)
    reference = predict(fresh, state, "pick up block")
    assert_current(actual, h, reference)
    assert actual.pointer == first.pointer
    assert actual.scales_pointer == first.scales_pointer
    assert h.pipe._lang_emb.copies == 1
    assert all(old is new for old, new in zip(graphs, graph_objects(h.pipe)))
    assert h.calls == before, "Same-shape state refresh must not calibrate or recapture"


def test_direct_frontend_infer_refreshes_state(frontend_factory):
    h, fresh = frontend_factory(), frontend_factory()
    predict(h, np.array([0.0]), "pick up block")
    state = np.array([0.75])
    actual = h.pipe.infer({"images": [h.image], "state": state})["actions"]
    assert_current(actual, h, predict(fresh, state, "pick up block"))


@pytest.mark.parametrize("changed", [0.0, 0.001, None])
def test_unchanged_or_omitted_state_reuses_conditioning(frontend_factory, changed):
    h = frontend_factory()
    first = predict(h, np.array([0.0]), "pick up block")
    before, graphs = h.calls.copy(), graph_objects(h.pipe)
    lookups = h.pipe.embedding_weight.lookups
    state = None if changed is None else np.array([changed])
    actual = predict(h, state)
    assert_current(actual, h, first)
    assert actual.pointer == first.pointer
    assert all(old is new for old, new in zip(graphs, graph_objects(h.pipe)))
    assert h.calls == before
    assert h.pipe.embedding_weight.lookups == lookups
    assert h.pipe._lang_emb.copies == 0


@pytest.mark.parametrize("old,new", [(-0.75, 0.0), (0.0, -0.75)])
def test_raw_length_change_with_equal_padded_length_reuses_storage(frontend_factory, old, new):
    h, fresh = frontend_factory(), frontend_factory()
    old, new = np.array([old]), np.array([new])
    lengths = [len(h.pipe._tokenize_prefix("pick up block", state)) for state in (old, new)]
    assert sorted(lengths) == [33, 34]
    first = predict(h, old, "pick up block")
    before, graphs = h.calls.copy(), graph_objects(h.pipe)
    actual = predict(h, new)
    assert_current(actual, h, predict(fresh, new, "pick up block"))
    assert actual.pointer == first.pointer
    assert all(old is new for old, new in zip(graphs, graph_objects(h.pipe)))
    assert h.calls == before


@pytest.mark.parametrize("old,new", [(0.0, -0.75), (-0.75, 0.0)])
@pytest.mark.parametrize("decode_graph", [False, True])
def test_shape_change_rebuilds_and_calibrates_before_replay(
        frontend_factory, old, new, decode_graph):
    h, fresh = frontend_factory(decode_graph), frontend_factory(decode_graph)
    predict(h, np.full(2, old), "pick up block")
    old_length, before, graphs = h.pipe.Se, h.calls.copy(), graph_objects(h.pipe)
    actual = predict(h, np.full(2, new))
    reference = predict(fresh, np.full(2, new), "pick up block")
    assert_current(actual, h, reference)
    assert h.pipe.Se != old_length
    assert h.pipe.prefill_len == reference.prefill_length
    assert h.calls["vision_capture"] > before["vision_capture"]
    assert h.calls["prefill_capture"] > before["prefill_capture"]
    assert h.calls["real_calibration"] == before["real_calibration"] + 1
    assert h.pipe._siglip_graph is not graphs[0]
    assert h.pipe._prefill_graph is not graphs[1]
    if decode_graph:
        assert h.pipe._decode_graph is not graphs[2]
        assert h.pipe._decode_graph.prefill_length == h.pipe.Se
    else:
        assert h.calls["decode_capture"] == 0


def test_capacity_error_preserves_previous_conditioning(frontend_factory):
    h = frontend_factory()
    first = predict(h, np.array([0.0]), "pick up block")
    before = h.calls.copy()
    lookups = h.pipe.embedding_weight.lookups
    with pytest.raises(ValueError, match="(?i)prefix|length|capacity"):
        predict(h, np.zeros(300))
    assert h.calls == before
    assert h.pipe.embedding_weight.lookups == lookups
    assert_current(predict(h), h, first)


def test_changed_task_explicitly_without_state_clears_old_state(frontend_factory):
    h, fresh = frontend_factory(), frontend_factory()
    predict(h, np.array([0.75]), "pick up block")
    actual = predict(h, prompt="place block")
    assert_current(actual, h, predict(fresh, prompt="place block"))


def test_pretokenized_prefix_is_preserved(frontend_factory):
    h = frontend_factory()
    h.pipe.set_prompt(np.array([1, 20, 30]))
    first = h.pipe.infer({"images": [h.image]})["actions"]
    before = h.calls.copy()
    actual = h.pipe.infer({"images": [h.image], "state": np.array([0.75])})["actions"]
    assert_current(actual, h, first)
    assert actual.pointer == first.pointer
    assert h.calls == before


def test_initial_wrapper_calibration_uses_current_prefix_once(frontend_factory):
    h = frontend_factory()
    predict(h, np.array([0.0]), "pick up block")
    assert not h.model._needs_real_data_calibration
    assert h.pipe._real_data_calibrated
    assert h.calls["real_calibration"] == 1
    before = h.calls.copy()
    predict(h, np.array([0.0]))
    assert h.calls == before


def test_first_observation_mapping_state_reaches_calibration(frontend_factory):
    h, fresh = frontend_factory(), frontend_factory()
    state = np.array([0.75])
    actual = h.model.predict({"images": [h.image], "state": state}, prompt="pick up block")
    assert_current(actual, h, predict(fresh, state, "pick up block"))
    assert h.calls["real_calibration"] == 1
    assert not h.model._needs_real_data_calibration


def test_repeated_state_changes_and_return_to_initial_state(frontend_factory):
    h = frontend_factory()
    first = predict(h, np.array([0.0]), "pick up block")
    before, graphs = h.calls.copy(), graph_objects(h.pipe)
    for copies, value in enumerate([0.75, 0.0, 0.75], start=1):
        state = np.array([value])
        fresh = frontend_factory()
        actual = predict(h, state)
        assert_current(actual, h, predict(fresh, state, "pick up block"))
        assert actual.pointer == first.pointer
        assert actual.scales_pointer == first.scales_pointer
        assert h.pipe._lang_emb.copies == copies
        assert h.calls == before
        assert all(old is new for old, new in zip(graphs, graph_objects(h.pipe)))
        predict(h, state)
        assert h.pipe._lang_emb.copies == copies, "Repeated tokens must be a no-op"


@pytest.mark.parametrize("task", ["move up block", "place block"])
def test_explicit_prompt_rebuild_invalidates_real_calibration(frontend_factory, task):
    h, fresh = frontend_factory(), frontend_factory()
    state = np.array([0.0])
    predict(h, state, "pick up block")
    before = h.calls["real_calibration"]
    h.pipe.set_prompt(task, state=state)
    assert not h.pipe._real_data_calibrated
    actual = h.pipe.infer({"images": [h.image], "state": state})["actions"]
    assert_current(actual, h, predict(fresh, state, task))
    assert h.calls["real_calibration"] == before + 1


def test_capacity_check_includes_even_padding(frontend_factory):
    h = frontend_factory()
    first = predict(h, np.array([0.0]), "pick")
    before, graphs = h.calls.copy(), graph_objects(h.pipe)
    lookups = h.pipe.embedding_weight.lookups
    h.pipe.Se_max = 35
    # 33 language tokens + 2 image tokens fit, but even padding requires 36.
    with pytest.raises(ValueError, match="(?i)prefix|length|capacity"):
        predict(h, np.array([-0.75]), "pick up block")
    assert h.calls == before
    assert h.pipe.embedding_weight.lookups == lookups
    assert all(old is new for old, new in zip(graphs, graph_objects(h.pipe)))
    assert_current(predict(h, np.array([0.0])), h, first)
