"""DROID decoding optimization must not prevent other checkpoint layouts loading."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/groot_n17'))
from groot_n17_iwm import fast_decode


def test_finetune_without_droid_keeps_native_decoder():
    decode = Mock()
    processor = SimpleNamespace(modality_configs={'custom_robot': {}}, decode_action=decode)
    with patch.object(fast_decode, 'FastOXEDecoder') as constructor:
        assert fast_decode.install_fast_decode(SimpleNamespace(processor=processor)) is None
    constructor.assert_not_called()
    assert processor.decode_action is decode
    assert not hasattr(processor, '_instinctflash_fast_decoder')


def test_existing_droid_decoder_is_reused():
    decoder = Mock()
    processor = SimpleNamespace(modality_configs={fast_decode.OXE_DROID: {}},
                                decode_action=decoder, _instinctflash_fast_decoder=decoder)
    with patch.object(fast_decode, 'FastOXEDecoder') as constructor:
        assert fast_decode.install_fast_decode(SimpleNamespace(processor=processor)) is decoder
    constructor.assert_not_called()
