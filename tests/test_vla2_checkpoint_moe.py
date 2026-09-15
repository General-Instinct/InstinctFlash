"""Header validation and lazy layer loading for direct VLA2 MoE construction."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'serving'))
from flash_rt.models.vla2.checkpoint_moe import checkpoint_moe_layers,MOE_PREFIX,MOE_SHAPES,MOE_LAYERS


def fixture(bad=None):
    reads=[];opened=[];closed=[]
    keys={f'{MOE_PREFIX}{i}.mlp.{s}':(s,shape) for i in range(MOE_LAYERS) for s,shape in MOE_SHAPES.items()}
    class Reader:
        def get_slice(self,key):
            shape=keys[key][1]
            return SimpleNamespace(get_shape=lambda: (1,) if key==bad else shape)
        def get_tensor(self,key):reads.append(key);return ('source_tensor',key)
    @contextmanager
    def safe_open(path,**kwargs):
        opened.append(path)
        try:yield Reader()
        finally:closed.append(path)
    return keys,reads,opened,closed,safe_open


def test_headers_validate_before_lazy_tensor_materialization():
    keys,reads,opened,closed,reader=fixture()
    with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'safetensors':SimpleNamespace(safe_open=reader)}):
        weight_map={k:f'model-{i%2}.safetensors' for i,k in enumerate(keys)}
        (Path(td)/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weight_map}))
        with checkpoint_moe_layers(td) as layers:
            assert len(layers)==36 and not reads and len(opened)==2
            layer=next(iter(layers))
            assert set(layer)==set(MOE_SHAPES) and len(reads)==5
            assert all(key.startswith(MOE_PREFIX+'0.mlp.') for key in reads)
            assert not closed
        assert sorted(opened)==sorted(closed)


def test_wrong_geometry_is_rejected_without_loading_weights():
    bad=f'{MOE_PREFIX}35.mlp.experts.down_proj'
    keys,reads,opened,closed,reader=fixture(bad)
    with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'safetensors':SimpleNamespace(safe_open=reader)}):
        try:
            with checkpoint_moe_layers(td):raise AssertionError('invalid geometry accepted')
        except ValueError as e:assert bad in str(e) and 'expected' in str(e)
        assert not reads and opened==closed


if __name__=='__main__':
    from run_tests import run_module_tests
    raise SystemExit(run_module_tests(globals()))
