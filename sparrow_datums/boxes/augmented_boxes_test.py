import doctest

import numpy as np
import numpy.typing as npt
import pytest

from ..types import PType
from . import augmented_boxes
from .augmented_boxes import AugmentedBoxes


def test_docstring_example():
    result = doctest.testmod(augmented_boxes)
    assert result.attempted > 0
    assert result.failed == 0


def test_augmented_boxes_with_6d_array_succeeds():
    x: npt.NDArray[np.float64] = np.ones((5, 6))
    boxes = AugmentedBoxes(x, ptype=PType.relative_tlwh)
    assert boxes.ptype == PType.relative_tlwh


def test_wrong_shape_throws_value_error():
    x: npt.NDArray[np.float64] = np.ones((5, 4))
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_non_integer_labels_throws_value_error():
    x: npt.NDArray[np.float64] = np.ones((5, 6))
    x[..., -1] += np.random.uniform(size=5)
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_invalid_scores_range_throws_value_error():
    x: npt.NDArray[np.float64] = np.ones((5, 6))
    x[..., -2] += 1
    with pytest.raises(ValueError):
        AugmentedBoxes(x)


def test_to_relative_moves_boxes_to_0_1():
    image_size = 512
    x: npt.NDArray[np.float64] = np.random.uniform(size=(10, 6))
    x[..., :4] *= image_size
    x[..., -1] = np.round(x[..., -1])
    boxes_a = AugmentedBoxes(
        x,
        PType.absolute_tlwh,
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
        PType.relative_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_absolute()
    assert boxes_b.is_absolute
    assert (boxes_b.array[..., :4] < 1).mean() < 0.1
    boxes_c = boxes_b.to_absolute()
    assert boxes_c.is_absolute


def test_to_tlbr_moves_boxes_to_tlbr():
    x: npt.NDArray[np.float64] = np.concatenate([np.ones((5, 2)), np.zeros((5, 4))], -1)
    boxes = AugmentedBoxes(x, PType.relative_tlwh).to_tlbr()
    np.testing.assert_equal(boxes.array[..., :4], 1)
    np.testing.assert_equal(boxes.array[..., 4:], 0)


def test_to_tlwh_moves_boxes_to_tlwh():
    result: npt.NDArray[np.float64] = np.concatenate(
        [np.ones((5, 2)), np.zeros((5, 2))], -1
    )
    boxes = AugmentedBoxes(np.ones((5, 6)), PType.relative_tlbr).to_tlwh()
    np.testing.assert_equal(boxes.array[..., :4], result)
    np.testing.assert_equal(boxes.array[..., 4:], 1)


def test_scores_and_labels_attributes():
    boxes_array = np.ones((6, 4))
    scores = np.random.uniform(size=6)
    labels = np.random.randint(10, size=6)
    boxes = AugmentedBoxes(
        np.concatenate([boxes_array, scores[:, None], labels[:, None]], -1)
    )
    np.testing.assert_almost_equal(scores, boxes.scores)
    np.testing.assert_almost_equal(labels, boxes.labels)


def test_multi_dimensional_label_names():
    label_names = ["a", "b", "c", "d", "e", "f"]
    boxes = np.ones((6, 4))
    scores: npt.NDArray[np.float64] = np.random.uniform(size=6)
    labels = np.arange(6)
    x: npt.NDArray[np.float64] = np.concatenate(
        [boxes, scores[:, None], labels[:, None]], -1
    )
    boxes = AugmentedBoxes(x.reshape(2, 3, 6))
    np.testing.assert_equal(
        boxes.names(label_names), np.array(label_names).reshape(2, 3)
    )
