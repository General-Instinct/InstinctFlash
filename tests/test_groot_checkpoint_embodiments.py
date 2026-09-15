"""FP8 must select checkpoint-owned embodiment weights, including fine-tunes."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'serving'))
from flash_rt.frontends.torch.groot_n17_thor import GrootN17TorchFrontendThor
from flash_rt.models.groot_n17.embodiments import checkpoint_embodiment_slot


@pytest.mark.parametrize('tag,slot', [
    ('oxe_droid_relative_eef_relative_joint', 7),
    ('xdof_relative_eef_relative_joint_subtask', 27),
    ('custom_finetune', 3),
])
def test_constructor_uses_checkpoint_mapping(tmp_path, tag, slot):
    (tmp_path / 'embodiment_id.json').write_text(json.dumps({tag: slot}))
    with patch.object(GrootN17TorchFrontendThor, '_load_weights'):
        frontend = GrootN17TorchFrontendThor(
            str(tmp_path), embodiment_tag=tag, load_strided_fmha=False)
    assert frontend._embodiment_id == slot


@pytest.mark.parametrize('slot', [True, -1, 1.5, '24', None])
def test_invalid_slot_refused(tmp_path, slot):
    (tmp_path / 'embodiment_id.json').write_text(json.dumps({'custom': slot}))
    with pytest.raises(ValueError, match='invalid embodiment slot'):
        checkpoint_embodiment_slot(str(tmp_path), 'custom')


def test_no_silent_builtin_mapping_fallback(tmp_path):
    tag = 'oxe_droid_relative_eef_relative_joint'
    with pytest.raises(FileNotFoundError):
        checkpoint_embodiment_slot(str(tmp_path), tag)
    (tmp_path / 'embodiment_id.json').write_text('{}')
    with pytest.raises(ValueError, match='does not declare'):
        checkpoint_embodiment_slot(str(tmp_path), tag)
