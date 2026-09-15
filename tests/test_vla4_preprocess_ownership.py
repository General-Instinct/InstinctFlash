"""Multiple runtimes must restore native preprocessing even when closed out of order."""
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/lingbot_vla'))
from lingbot_vla_iwm.image_preprocess import GPUImagePreprocess, _upstream_prepare


def test_out_of_order_close_unlinks_dormant_wrappers_and_releases_storage():
    def native(*args):
        return 'native'
    def wrapper_a(*args):
        return 'a'
    def wrapper_b(*args):
        return 'b'
    releases = []
    a = GPUImagePreprocess(SimpleNamespace(), native, wrapper_a, 'cuda:0')
    b = GPUImagePreprocess(SimpleNamespace(), wrapper_a, wrapper_b, 'cuda:1')
    wrapper_a._ifl_preprocess_owner = a
    wrapper_b._ifl_preprocess_owner = b
    wrapper_a._ifl_release_storage = lambda: releases.append('a')
    wrapper_b._ifl_release_storage = lambda: releases.append('b')
    modules = {name: ModuleType(name) for name in ('lingbotvla', 'lingbotvla.data',
               'lingbotvla.data.vla_data', 'lingbotvla.data.vla_data.utils')}
    utils = modules['lingbotvla.data.vla_data.utils']
    utils.prepare_images = wrapper_b
    assert _upstream_prepare(wrapper_b) is native
    with patch.dict(sys.modules, modules):
        a.close()
        assert utils.prepare_images is wrapper_b
        assert a.rejected and a.server is None
        b.close()
        assert utils.prepare_images is native
    assert releases == ['a', 'b']
