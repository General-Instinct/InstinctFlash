"""CPU lifecycle tests; these do not establish CUDA fit or CUDA numerical parity."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from instinctflash.runtime import lingbot_install as install
from instinctflash.runtime.lingbot_residency import needs_prompt_encoder_staging


@pytest.fixture(autouse=True)
def clean_staging_tokens():
    install._PROMPT_ENCODER_STAGING.__dict__.clear()
    yield
    install._PROMPT_ENCODER_STAGING.__dict__.clear()


@pytest.mark.parametrize("capability,memory,expected", [
    ((8, 9), 24 << 30, True), ((12, 0), 32 << 30, True),
    ((8, 9), 48 << 30, False), ((11, 0), 128 << 30, False),
    ((9, 0), 80 << 30, False), ((8, 9), 0, False),
])
def test_memory_policy(capability, memory, expected):
    assert needs_prompt_encoder_staging(capability, memory) is expected


def fake_cuda(monkeypatch, events):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(major=8, minor=9, total_memory=24 << 30,
                                        name="NVIDIA GeForce RTX 4090", uuid="CPU-lifecycle-test-only"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))


def test_allocator_elision_is_instance_and_call_scoped(monkeypatch):
    events = []
    native_empty_cache = lambda: events.append("release")
    monkeypatch.setattr(torch.cuda, "empty_cache", native_empty_cache)
    module = SimpleNamespace(torch=torch)

    class Server:
        def __init__(self, fail=False):
            if fail:
                raise RuntimeError("constructor failure")

        def _reset(self, prompt=None):
            module.torch.cuda.empty_cache()

        def _infer(self, obs, frame_st_id=0):
            module.torch.cuda.empty_cache()
            if obs == "fail":
                raise RuntimeError("action failure")

        def _compute_kv_cache(self, obs):
            module.torch.cuda.empty_cache()

    install.install_allocator_churn_elision(module, Server)
    optimized, native = Server(), Server()
    assert torch.cuda.empty_cache is native_empty_cache
    for method, value in (("_reset", "prompt"), ("_infer", {}), ("_compute_kv_cache", {})):
        getattr(optimized, method)(value)
        assert events == []
        getattr(native, method)(value)
        assert events == ["release"]
        events.clear()
    with pytest.raises(RuntimeError, match="action failure"):
        optimized._infer("fail")
    module.torch.cuda.empty_cache()
    assert events == ["release"]  # failed calls must restore dispatch context

    install.install_allocator_churn_elision(module, Server)
    with pytest.raises(RuntimeError, match="constructor failure"):
        Server(fail=True)
    events.clear()
    Server()._infer({})
    assert events == ["release"]  # a failed constructor consumes its token


def fake_prompt_stack(monkeypatch, events):
    """Use real tiny CPU tensors and fake placement; never execute CUDA."""
    fake_cuda(monkeypatch, events)

    class Encoder(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16))
            self.location = str(device)

        def to(self, device):
            events.append(("encoder_to", str(device)))
            self.location = str(device)
            return self

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_k = self.to_v = self.norm_k = torch.nn.Identity()
            self.heads = 1

        def forward(self, *args, **kwargs):
            return args[0]

    class TextProjection(torch.nn.Module):
        def forward(self, value):
            events.append("cross_project")
            return value * 2

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([SimpleBlock()])
            self.condition_embedder = SimpleNamespace(text_embedder=TextProjection())
            self.history = ["old episode"]
            self.pool = None

        def clear_cache(self, name):
            events.append("clear_history")
            self.history.clear()
            self.pool = None

        def create_empty_cache(self, cache_name, attn_window, latent_tokens, action_tokens,
                               device, dtype, batch_size):
            events.append("allocate_full_history")
            self.pool = (cache_name, attn_window, latent_tokens, action_tokens,
                         device, dtype, batch_size)

        def forward(self, *args, **kwargs):
            return None

    class SimpleBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn2 = Attention()

    class VAE:
        def __init__(self):
            self.history = ["old episode"]
            self.vae = SimpleNamespace(decoder=torch.nn.Linear(1, 1, bias=False))

        def clear_cache(self):
            events.append("clear_vae")
            self.history.clear()

    module = SimpleNamespace()

    def load_text_encoder(path, torch_dtype, torch_device):
        events.append(("load_encoder", str(torch_device)))
        return Encoder(torch_device)

    module.load_text_encoder = load_text_encoder

    class Server:
        def __init__(self, job_config):
            self.job_config = job_config
            self.device, self.dtype, self.cache_name = "cuda:0", torch.bfloat16, "pos"
            self.text_encoder = module.load_text_encoder("weights", self.dtype, self.device)
            if job_config.fail_load:
                raise RuntimeError("load failure")
            self.transformer = Model()
            self.streaming_vae, self.streaming_vae_half = VAE(), VAE()
            self.init_latent = "old latent"

        def _reset(self, prompt=None):
            events.append(("native_reset", prompt, self.text_encoder.location))
            self.transformer.clear_cache(self.cache_name)
            self.streaming_vae.clear_cache()
            self.streaming_vae_half.clear_cache()
            self.frame_st_id = 0
            self.init_latent = None
            self.use_cfg = self.job_config.guidance_scale > 1
            for _ in range(getattr(self.job_config, "cache_allocations", 1)):
                self.transformer.create_empty_cache(
                    self.cache_name, 72, 240, 32, device=self.device,
                    dtype=self.dtype, batch_size=2 if self.use_cfg else 1)
            self.prompt_embeds = torch.tensor([[[len(prompt or "")]]], dtype=self.dtype)
            self.negative_prompt_embeds = torch.zeros_like(self.prompt_embeds)
            if prompt == "fail":
                self.transformer.history.append("partial new state")
                raise RuntimeError("prompt failure")

    model_module = ModuleType("modules.model")
    model_module.WanAttention = Attention
    model_module.WanTransformer3DModel = Model
    package = ModuleType("modules")
    package.model = model_module
    monkeypatch.setitem(sys.modules, "modules", package)
    monkeypatch.setitem(sys.modules, "modules.model", model_module)
    config = SimpleNamespace(local_rank=0, guidance_scale=5, fail_load=False)
    return module, Server, config


@pytest.mark.parametrize("capability,name,memory", [
    ((8, 9), "NVIDIA GeForce RTX 4090", 24 << 30),
    ((12, 0), "NVIDIA GeForce RTX 5090", 32 << 30),
])
def test_native_reference_wrapper_preserves_native_calls_and_records_residency(monkeypatch, tmp_path, capability, name, memory):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(major=capability[0], minor=capability[1], total_memory=memory,
                                        name=name, uuid="CPU-lifecycle-test-only"))
    module.VA_Server = Server
    source = tmp_path / "cpu_fixture_server.py"
    source.write_text("# CPU-only native-server lifecycle fixture\n")
    module.__file__ = str(source)
    loader = module.load_text_encoder
    server, receipt = install.build_native_reference_server(module, config)
    assert module.VA_Server is Server and module.load_text_encoder is loader
    assert server.text_encoder.location == "cpu"
    assert receipt["successful_resets"] == 0
    assert receipt["device"]["capability"] == list(capability)
    assert receipt["removed_decoder_count"] == 2
    assert receipt["removed_decoder_bytes"] == 8
    assert not hasattr(Server, "_iwm_prompt_encoder_load_staging_installed")
    assert not hasattr(server.transformer, "populate_cross_cache")
    for prompt in ("first", "second"):
        server._reset(prompt)
    assert receipt["successful_resets"] == 2
    assert receipt["staged_encoder_bytes"] == 4
    assert receipt["last_full_history_allocation"] == {
        "cache_name": "pos", "attn_window": 72, "latent_tokens": 240,
        "action_tokens": 32, "device": "cuda:0", "dtype": "torch.bfloat16", "batch_size": 2}
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "native_reset"] == ["first", "second"]
    assert "cross_project" not in events
    # A subsequent ordinary native constructor retains its decoder and GPU T5 residency.
    untouched = Server(config)
    assert untouched.text_encoder.location == "cuda:0"
    assert isinstance(untouched.streaming_vae.vae.decoder, torch.nn.Linear)


def test_native_reference_wrapper_rejects_different_actual_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device:
                        SimpleNamespace(major=8, minor=9, total_memory=48 << 30,
                                        name="NVIDIA L40S", uuid="CPU-test-only"))
    with pytest.raises(RuntimeError, match="only on RTX 4090"):
        install.build_native_reference_server(SimpleNamespace(), SimpleNamespace())


def test_staging_binds_at_load_and_repeats_across_episodes(monkeypatch):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    install.install_prompt_encoder_staging(module, Server)
    install.install_conditioning_prefill(module, Server)
    staged = Server(config)
    native = Server(config)
    assert staged.text_encoder.location == "cpu"
    assert native.text_encoder.location == "cuda:0"
    assert not native._iwm_prompt_encoder_staged_decision
    events.clear()
    staged._reset("first")
    assert events.index("clear_history") < events.index(("encoder_to", "cuda:0"))
    assert events.index(("encoder_to", "cpu")) < events.index("cross_project")
    assert events.index(("encoder_to", "cpu")) < events.index("allocate_full_history")
    assert events.index("allocate_full_history") < events.index("cross_project")
    assert staged.transformer.pool == ("pos", 72, 240, 32, "cuda:0", torch.bfloat16, 2)
    assert "create_empty_cache" not in staged.transformer.__dict__
    assert staged.text_encoder.location == "cpu"
    assert staged._iwm_prompt_encoder_staged_bytes == 4
    key, value = staged.transformer.blocks[0].attn2._iwm_cross_kv
    assert torch.equal(key, torch.tensor([[[[10.]]], [[[0.]]]], dtype=torch.bfloat16))
    assert torch.equal(key, value)

    # History from completed 4/8-frame commits must be released before the next T5 load.
    staged.transformer.history.extend([4, 8])
    staged.streaming_vae.history.extend([4, 8])
    staged.streaming_vae_half.history.extend([4, 8])
    staged.init_latent = "last episode latent"
    config.guidance_scale = 1
    events.clear()
    staged._reset("second episode")
    assert events.index("clear_history") < events.index(("encoder_to", "cuda:0"))
    assert not staged.transformer.history and not staged.streaming_vae.history
    assert not staged.streaming_vae_half.history and staged.init_latent is None
    key, _ = staged.transformer.blocks[0].attn2._iwm_cross_kv
    assert key.shape == (1, 1, 1, 1)  # changed CFG resets cross-KV batch shape
    assert key.item() == 28
    assert staged.transformer.pool == ("pos", 72, 240, 32, "cuda:0", torch.bfloat16, 1)


def test_failed_prompt_reset_releases_encoder_and_partial_history(monkeypatch):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    install.install_prompt_encoder_staging(module, Server)
    install.install_conditioning_prefill(module, Server)
    server = Server(config)
    with pytest.raises(RuntimeError, match="prompt failure"):
        server._reset("fail")
    assert server.text_encoder.location == "cpu"
    assert not server.transformer.history
    assert server.transformer.pool is None
    assert "create_empty_cache" not in server.transformer.__dict__
    assert server.prompt_embeds is None and server.negative_prompt_embeds is None
    assert not server.transformer.cross_cache_populated()
    server._reset("recovered")
    assert server.text_encoder.location == "cpu"
    assert server.transformer.cross_cache_populated()


def test_failed_constructor_cannot_stage_a_later_unarmed_server(monkeypatch):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    install.install_prompt_encoder_staging(module, Server)
    install.install_conditioning_prefill(module, Server)
    config.fail_load = True
    with pytest.raises(RuntimeError, match="load failure"):
        Server(config)
    assert install._PROMPT_ENCODER_STAGING.building is None
    config.fail_load = False
    next_server = Server(config)
    assert next_server.text_encoder.location == "cuda:0"
    assert not next_server._iwm_prompt_encoder_staged_decision


@pytest.mark.parametrize("allocations", [0, 2])
def test_changed_reset_allocation_contract_fails_closed(monkeypatch, allocations):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    install.install_prompt_encoder_staging(module, Server)
    install.install_conditioning_prefill(module, Server)
    server = Server(config)
    config.cache_allocations = allocations
    with pytest.raises(RuntimeError, match="exactly one native KV allocation"):
        server._reset("episode")
    assert "allocate_full_history" not in events
    assert server.text_encoder.location == "cpu"
    assert server.prompt_embeds is None and server.transformer.pool is None
    assert "create_empty_cache" not in server.transformer.__dict__


def test_deferred_allocation_failure_restores_original_method(monkeypatch):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    install.install_prompt_encoder_staging(module, Server)
    install.install_conditioning_prefill(module, Server)
    server = Server(config)

    def failed_allocation(*args, **kwargs):
        assert server.text_encoder.location == "cpu"
        raise RuntimeError("allocation failure")

    server.transformer.create_empty_cache = failed_allocation
    with pytest.raises(RuntimeError, match="allocation failure"):
        server._reset("episode")
    assert server.transformer.create_empty_cache is failed_allocation
    assert server.prompt_embeds is None and server.transformer.pool is None
    assert server.text_encoder.location == "cpu"


def test_prefill_reinstall_does_not_duplicate_reset_staging(monkeypatch):
    events = []
    module, Server, config = fake_prompt_stack(monkeypatch, events)
    install.install_conditioning_prefill(module, Server)
    first = Server(config)
    install.install_prompt_encoder_staging(module, Server)
    install.install_conditioning_prefill(module, Server)
    second = Server(config)
    for server in (first, second):
        events.clear()
        server._reset("episode")
        assert events.count(("encoder_to", "cuda:0")) == 1
        assert events.count(("encoder_to", "cpu")) == 1
        assert events.count("cross_project") == 1


def test_staging_loader_shape_guard_does_not_modify_constructor():
    class Server:
        pass
    original = Server.__init__
    with pytest.raises(RuntimeError, match="must accept torch_device"):
        install._install_prompt_encoder_load_staging(SimpleNamespace(), Server)
    assert Server.__init__ is original
