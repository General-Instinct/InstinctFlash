"""The padding exception must never hide active action or video input changes."""
import numpy as np
import pytest

from export_control import compare_inputs


def arrays():
    return {**{key: np.zeros((1, 4 + 33 * 64), np.float32)
               for key in ('reference', 'initial_noise', 'preserve_mask')},
            **{key: np.zeros((32, 8), np.float32)
               for key in ('measured_action', 'model_measured_action')}}


def test_only_declared_noise_and_mask_padding_may_differ():
    left, right = arrays(), arrays()
    for key in ('initial_noise', 'preserve_mask'):
        right[key][:, 4:].reshape(1, 33, 64)[:, :, 8:] = 1
    compare_inputs(left, right, native_padding=True, offset=4)
    with pytest.raises(AssertionError):
        compare_inputs(left, right, native_padding=False, offset=4)


@pytest.mark.parametrize('field,coordinate', [
    ('initial_noise', 0), ('initial_noise', 4), ('initial_noise', 4 + 64 + 7),
    ('preserve_mask', 4 + 32 * 64), ('reference', 4 + 8),
])
def test_other_input_changes_are_rejected(field, coordinate):
    left, right = arrays(), arrays()
    right[field][0, coordinate] = 1
    with pytest.raises(AssertionError):
        compare_inputs(left, right, native_padding=True, offset=4)
