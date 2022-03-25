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


def test_score_alias_is_float():
    box = SingleAugmentedBox(np.ones(6))
    assert isinstance(box.score, float)
    assert box.score == 1


def test_label_alias_is_int():
    box = SingleAugmentedBox(np.ones(6))
    assert isinstance(box.label, int)
    assert box.label == 1


def test_to_single_box_preserves_as_much_data_as_possible():
    single_augmented_box = SingleAugmentedBox(np.array([0, 0, 1, 1, 0.4, 7]))
    single_box = single_augmented_box.to_single_box()
    np.testing.assert_equal(single_augmented_box.array[:4], single_box.array)
    assert single_augmented_box.ptype == single_box.ptype
