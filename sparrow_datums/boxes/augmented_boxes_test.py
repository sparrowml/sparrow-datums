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
    x[..., -1] += np.random.uniform(size=5)
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_invalid_scores_range_throws_value_error():
    x = np.ones((5, 6))
    x[..., -2] += 1
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_to_relative_moves_boxes_to_0_1():
    image_size = 512
    x = np.random.uniform(size=(10, 6))
    x[..., :4] *= image_size
    x[..., -1] = np.round(x[..., -1])
    boxes_a = AugmentedBoxes(
        x,
        BoxType.absolute_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_relative()
    assert boxes_b.is_relative
    np.testing.assert_array_less(boxes_b.array, 1.01)
    boxes_c = boxes_b.to_relative()
    assert boxes_c.is_relative


def test_to_absolute_moves_boxes_to_0_image_size():
    image_size = 512
    x = np.random.uniform(size=(10, 6))
    x[..., -1] = np.round(x[..., -1])
    boxes_a = AugmentedBoxes(
        x,
        BoxType.relative_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_absolute()
    assert boxes_b.is_absolute
    assert (boxes_b.array[..., :4] < 1).mean() < 0.1
    boxes_c = boxes_b.to_absolute()
    assert boxes_c.is_absolute


def test_to_tlbr_moves_boxes_to_tlbr():
    x = np.concatenate([np.ones((5, 2)), np.zeros((5, 4))], -1)
    boxes = AugmentedBoxes(x, BoxType.relative_tlwh).to_tlbr()
    np.testing.assert_equal(boxes.array[..., :4], 1)
    np.testing.assert_equal(boxes.array[..., 4:], 0)


def test_to_tlwh_moves_boxes_to_tlwh():
    result = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = AugmentedBoxes(np.ones((5, 6)), BoxType.relative_tlbr).to_tlwh()
    np.testing.assert_equal(boxes.array[..., :4], result)
    np.testing.assert_equal(boxes.array[..., 4:], 1)


def test_scores_and_labels_attributes():
    boxes_array = np.ones((6, 4))
    scores = np.random.uniform(size=6)
    labels = np.random.randint(10, size=6)
    boxes = AugmentedBoxes(
        np.concatenate([boxes_array, scores[:, None], labels[:, None]], -1)
    )
    np.testing.assert_equal(scores, boxes.scores)
    np.testing.assert_equal(labels, boxes.labels)
