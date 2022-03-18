import numpy as np
import pytest

from .single_box import SingleAugmentedBox, SingleBox
from .types import BoxType


def test_single_box_conversion_creates_single_box():
    box_a = SingleBox(np.ones(4), BoxType.absolute_tlbr)
    box_b = box_a.to_tlwh()
    assert box_b.is_tlwh
    assert isinstance(box_b, SingleBox)


def test_single_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleBox(np.ones((2, 4)))
    with pytest.raises(ValueError):
        SingleBox(np.ones(3))


def test_single_augmented_box_conversion_creates_single_augmented_box():
    box_a = SingleAugmentedBox(np.ones(6), BoxType.absolute_tlwh)
    box_b = box_a.to_tlbr()
    assert box_b.is_tlbr
    assert isinstance(box_b, SingleAugmentedBox)


def test_single_augmented_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleAugmentedBox(np.ones((2, 6)))
    with pytest.raises(ValueError):
        SingleAugmentedBox(np.ones(4))
