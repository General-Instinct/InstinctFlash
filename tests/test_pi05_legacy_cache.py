import sys,unittest
from pathlib import Path
if __name__ != '__main__':
    import pytest
    pytest.importorskip('transformers', reason='pi05 cache integration requires Transformers')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/pi05_vla'))
import torch
from transformers.cache_utils import DynamicCache
from pi05_iwm.static_capture import _StaticKV
from pi05_iwm.surface import Pi05CacheBinder

class CacheCompatibilityTests(unittest.TestCase):
    def cache(self):
        cache=DynamicCache()
        cache.update(torch.arange(12.).reshape(1,1,3,4),torch.ones(1,1,3,4),0)
        return cache
    def test_runtime_cache_roundtrip_preserves_shapes_and_bytes(self):
        cache=self.cache();binder=Pi05CacheBinder();leaves,spec=binder.flatten(cache)
        restored=binder.unflatten([x.clone() for x in leaves],spec)
        self.assertEqual(len(list(restored)),len(list(cache)))
        for a,b in zip(list(cache)[0][:2],list(restored)[0][:2]):
            self.assertTrue(torch.equal(a,b))
    def test_legacy_indexing_exposes_only_live_prefix_after_refill(self):
        cache=self.cache();static=_StaticKV(cache,2)
        self.assertEqual(static[0][0].shape,(1,1,3,4))
        cache.update(torch.ones(1,1,1,4),torch.ones(1,1,1,4),0)
        # Use a same-shaped fresh prefix to test that indexing tracks new values.
        fresh=self.cache();list(fresh)[0][0].add_(10)
        static.refill(fresh)
        self.assertTrue(torch.equal(static[0][0],list(fresh)[0][0]))
        static.update(torch.zeros(1,1,2,4),torch.zeros(1,1,2,4),0)
        self.assertEqual(static[0][0].shape[2],3)
        self.assertTrue(torch.equal(static[0][0],list(fresh)[0][0]))
if __name__=='__main__':unittest.main()
