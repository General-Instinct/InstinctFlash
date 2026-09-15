from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from instinctflash.runtime.engine_backend import EngineBackend


@pytest.mark.parametrize('steps,guidance,packed', [(4, 3, True), (2, 3, True),
                                                  (4, 1, True), (4, 3, False)])
def test_actual_cosmos_build_gates_and_cleanup(steps, guidance, packed):
    loop = Mock()
    loop.declaration.return_value = {
        'precision': 'fp8', 'frontend': 'instinctflash/runtime/cosmos_droid.py',
        'steps': {'prefix': 1, 'action': steps},
        'guidance': {'action': ('cfg', guidance)}, 'evidence': 'test',
    }
    loop.backend_stats.return_value = {'fp8_recipe': {'projections': ['q'] if packed else []}}
    adapter = Mock()
    adapter.build_fp8.return_value = loop
    ckpt = SimpleNamespace(execution=SimpleNamespace(backbone='cosmos3_policy'))
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.get_device_capability', return_value=(11, 0)), \
         patch('instinctflash.runtime.engine_backend.requested_operating_point',
               return_value=({'prefix': 1, 'action': 4}, {'action': ('cfg', 3)}, 'test')):
        if steps == 4 and guidance == 3 and packed:
            backend = EngineBackend(adapter, ckpt, None)
            backend.predict({'state': []})
            loop.predict.assert_called_once()
            backend.close()
        else:
            with pytest.raises(RuntimeError):
                EngineBackend(adapter, ckpt, None)
            loop.predict.assert_not_called()
        loop.close.assert_called_once()
