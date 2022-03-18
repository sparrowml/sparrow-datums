import numpy as np
import pytest

from .frame_boxes import FrameAugmentedBoxes, FrameBoxes
from .types import BoxType


def test_frame_boxes_conversion_creates_more_frame_boxes():
    boxes_a = FrameBoxes(np.ones((2, 4)), BoxType.relative_tlbr)
    boxes_b = boxes_a.to_tlwh()
    assert isinstance(boxes_b, FrameBoxes)


def test_frame_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        FrameBoxes(np.ones(4))
    with pytest.raises(ValueError):
        FrameBoxes(np.ones((2, 3)))


def test_frame_augmented_boxes_conversion_creates_more_frame_augmented_boxes():
    boxes_a = FrameAugmentedBoxes(np.ones((2, 6)), type=BoxType.absolute_tlwh)
    boxes_b = boxes_a.to_tlbr()
    assert boxes_b.is_tlbr
    assert isinstance(boxes_b, FrameAugmentedBoxes)


def test_frame_augmented_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        FrameAugmentedBoxes(np.ones(6))
    with pytest.raises(ValueError):
        FrameAugmentedBoxes(np.ones((2, 4)))
