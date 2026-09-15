"""Build-declared VA admission must use the actual engine instance's schedule."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from instinctflash.runtime.engine_backend import EngineBackend


@pytest.mark.parametrize("actual_steps", [25, 2])
def test_actual_built_schedule_is_checked_and_mismatch_closed(actual_steps):
    checkpoint = SimpleNamespace(execution=SimpleNamespace(backbone="wan_va"))
    loop = Mock()
    loop.declaration.return_value = {
        "steps": {"video": actual_steps, "action": 50},
        "guidance": {"video": ("cfg", 5), "action": ("positive_only", 1)},
    }
    with patch("instinctflash.runtime.engine_backend.engine_available", return_value=(True, "test")), \
         patch("instinctflash.runtime.engine_backend.requested_operating_point",
               return_value=({"video": 25, "action": 50},
                             {"video": ("cfg", 5), "action": ("positive_only", 1)}, "test")), \
         patch("instinctflash.runtime.wan_va_engine_build.build_wan_va_engine_loop", return_value=loop):
        if actual_steps == 25:
            backend = EngineBackend(None, checkpoint, None)
            backend.predict({"obs": []}, executed_action="executed")
            loop.predict.assert_called_once_with({"obs": []}, executed_action="executed")
            backend.close()
        else:
            with pytest.raises(RuntimeError, match="requests video=25"):
                EngineBackend(None, checkpoint, None)
            loop.predict.assert_not_called()
        loop.close.assert_called_once()
