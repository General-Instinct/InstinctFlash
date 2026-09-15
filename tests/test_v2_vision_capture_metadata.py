"""Fixed vision metadata must be capture-safe without changing attention math."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/lingbot_vla_v2'))
from lingbot_vla_v2_iwm.prefix_capture import StaticVision


@pytest.mark.parametrize('implementation,host', [('sdpa', True), ('eager', True),
                                                ('flash_attention_2', False)])
def test_sequence_metadata_device_and_restore(implementation, host):
    sequence = Mock()
    original = (object(), object(), object(), object(), object())
    visual = SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(config=SimpleNamespace(
            _attn_implementation=implementation)))],
        preprcess_grid_thw=lambda **_: (torch.zeros(1), (torch.ones(1), torch.zeros(1)),
                                       sequence, [4], 4))
    expert = SimpleNamespace(qwenvl=SimpleNamespace(visual=visual),
                             embed_image=lambda *args: args)
    names = ('pos_embeds', 'position_embeddings', 'cu_seqlens',
             'visual_split_sizes', 'visual_max_seqlen')
    for name, value in zip(names, original):
        setattr(expert, name, value)
    driver = StaticVision(expert)
    driver(torch.zeros(4, 3), torch.tensor([[1, 2, 2]]))
    assert sequence.cpu.call_count == int(host)
    assert expert.cu_seqlens is (sequence.cpu.return_value if host else sequence)
    driver.close()
    assert all(getattr(expert, name) is value for name, value in zip(names, original))


def test_changed_grid_rejected_before_mutating_image_or_replaying():
    driver = object.__new__(StaticVision)
    driver._image = torch.zeros(4, 3)
    driver._grid = torch.tensor([[1, 2, 2]])
    driver.graph = Mock()
    with pytest.raises(RuntimeError, match='grid values changed'):
        driver(torch.ones(4, 3), torch.tensor([[1, 1, 4]]))
    assert torch.count_nonzero(driver._image) == 0
    driver.graph.replay.assert_not_called()
