import numpy as np
import pytest

from ..types import PType
from .single_augmented_box import SingleAugmentedBox


def test_single_augmented_box_conversion_creates_single_augmented_box():
    box_a = SingleAugmentedBox(np.ones(6), PType.absolute_tlwh)
    box_b = box_a.to_tlbr()
    assert box_b.is_tlbr
    assert isinstance(box_b, SingleAugmentedBox)


def test_single_augmented_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleAugmentedBox(np.ones((2, 6)))
    with pytest.raises(ValueError):
        SingleAugmentedBox(np.ones(4))