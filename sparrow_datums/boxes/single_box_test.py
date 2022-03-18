import numpy as np
import pytest

from .single_box import SingleAugmentedBox, SingleBox
from .types import BoxType


def test_valid_single_box_succeeds():
    box = SingleBox(np.ones(4), type=BoxType.absolute_tlwh)
    assert box.type == BoxType.absolute_tlwh


def test_single_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleBox(np.ones((2, 4)))
    with pytest.raises(ValueError):
        SingleBox(np.ones(3))


def test_single_box_to_tlbr_works():
    box = SingleBox(np.ones(4), type=BoxType.absolute_tlwh).to_tlbr()
    assert box.is_tlbr
    np.testing.assert_equal(box.array, np.array([1, 1, 2, 2]))


def test_valid_single_augmented_box_succeeds():
    box = SingleAugmentedBox(np.ones(6), type=BoxType.absolute_tlwh)
    assert box.type == BoxType.absolute_tlwh


def test_single_augmented_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleAugmentedBox(np.ones((2, 6)))
    with pytest.raises(ValueError):
        SingleAugmentedBox(np.ones(4))


def test_single_augmented_box_to_tlbr_works():
    box = SingleAugmentedBox(np.ones(6), type=BoxType.absolute_tlwh).to_tlbr()
    assert box.is_tlbr
    np.testing.assert_equal(box.array[..., :4], np.array([1, 1, 2, 2]))
    np.testing.assert_equal(box.array[..., 4:], np.ones(2))
