"""FP8 must select checkpoint-owned embodiment weights, including fine-tunes."""
import json

import pytest

from instinctflash.engines.groot_n17.weights import embodiment_id


@pytest.mark.parametrize('tag,slot', [
    ('oxe_droid_relative_eef_relative_joint', 7),
    ('xdof_relative_eef_relative_joint_subtask', 27),
    ('custom_finetune', 3),
])
def test_checkpoint_mapping_selects_the_slot(tmp_path, tag, slot):
    (tmp_path / 'embodiment_id.json').write_text(json.dumps({tag: slot}))
    assert embodiment_id(tmp_path, tag) == slot


@pytest.mark.parametrize('slot', [True, -1, 1.5, '24', None])
def test_invalid_slot_refused(tmp_path, slot):
    (tmp_path / 'embodiment_id.json').write_text(json.dumps({'custom': slot}))
    with pytest.raises(ValueError, match='invalid embodiment slot'):
        embodiment_id(tmp_path, 'custom')


def test_no_silent_builtin_mapping_fallback(tmp_path):
    tag = 'oxe_droid_relative_eef_relative_joint'
    with pytest.raises(FileNotFoundError):
        embodiment_id(tmp_path, tag)
    (tmp_path / 'embodiment_id.json').write_text('{}')
    with pytest.raises(ValueError, match='not declared'):
        embodiment_id(tmp_path, tag)
