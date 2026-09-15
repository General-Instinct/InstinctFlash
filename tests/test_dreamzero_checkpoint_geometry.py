import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/dreamzero'))
from dreamzero_iwm.adapter import DreamZeroAdapter


@pytest.mark.parametrize('tokens,width,layers', [(880,5120,40), (50,3072,30)])
def test_actual_checkpoint_controls_kv_geometry(tmp_path,tokens,width,layers):
    config = {'action_head_cfg': {'config': {'diffusion_model_cfg': {
        'frame_seqlen': tokens, 'dim': width, 'num_layers': layers}}}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    spec = DreamZeroAdapter().spec_for_checkpoint(SimpleNamespace(path=tmp_path))
    assert spec.streams[0].tokens_per_frame == tokens
    assert spec.notes['dit_width'] == width and spec.notes['dit_layers'] == layers
    assert spec.phase('video_action').nfe == 16


@pytest.mark.parametrize('tokens', [None, True, 0, -1, '880'])
def test_unknown_geometry_is_not_replaced_with_5b_guess(tmp_path,tokens):
    (tmp_path/'config.json').write_text(json.dumps({
        'action_head_cfg': {'config': {'diffusion_model_cfg': {'frame_seqlen': tokens}}}}))
    with pytest.raises(ValueError, match='frame_seqlen'):
        DreamZeroAdapter().spec_for_checkpoint(SimpleNamespace(path=tmp_path))
