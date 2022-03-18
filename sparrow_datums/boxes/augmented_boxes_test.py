import numpy as np
import pytest

from .augmented_boxes import AugmentedBoxes
from .types import BoxType


def test_augmented_boxes_with_6d_array_succeeds():
    x = np.ones((5, 6))
    boxes = AugmentedBoxes(x, type=BoxType.relative_tlwh)
    assert boxes.type == BoxType.relative_tlwh


def test_wrong_shape_throws_value_error():
    x = np.ones((5, 4))
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_non_integer_labels_throws_value_error():
    x = np.ones((5, 6))
    x[..., -2] += np.random.uniform(size=5)
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_invalid_scores_range_throws_value_error():
    x = np.ones((5, 6))
    x[..., -1] += 1
    with pytest.raises(ValueError):
        AugmentedBoxes(x)
