from types import SimpleNamespace
import pytest
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from prompt_contract import verify_plain_prompt

@pytest.mark.parametrize('overrides,accepted', [
    ({}, True), ({'format_prompt_as_json': True}, False),
    ({'append_viewpoint_info': False}, False),
    ({'append_duration_fps_timestamps': False}, False),
    ({'append_resolution_info': False}, False),
])
def test_actual_native_transform(overrides, accepted):
    pipeline = ActionTransformPipeline(tokenizer_config=None, max_action_dim=64, **overrides)
    service = SimpleNamespace(_transform=pipeline, cfg=SimpleNamespace(history_length=1, action_chunk_size=32))
    if accepted:
        assert verify_plain_prompt(service)['format_prompt_as_json'] is False
    else:
        with pytest.raises(ValueError):
            verify_plain_prompt(service)
