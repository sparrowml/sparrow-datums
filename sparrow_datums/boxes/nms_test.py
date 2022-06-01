import numpy as np

from ..types import PType
from .frame_augmented_boxes import FrameAugmentedBoxes
from .nms import non_max_suppression


def test_nms_filters_by_score():
    data = np.array([[0, 0, 1, 1, 0.4, 0], [1, 1, 1, 1, 0.6, 1]])
    boxes = FrameAugmentedBoxes(data, ptype=PType.absolute_tlwh)
    filtered_boxes = non_max_suppression(boxes)
    assert len(filtered_boxes) == 1


def test_nms_removes_overlapping_boxes_by_class():
    data = np.array(
        [
            [0, 0, 1, 1, 0.4, 0],
            [0.1, 0.1, 1, 1, 0.65, 0],
            [1, 1, 1, 1, 0.7, 1],
            [1.1, 1.1, 1, 1, 0.6, 1],
        ]
    )
    boxes = FrameAugmentedBoxes(data, ptype=PType.absolute_tlwh)
    filtered_boxes = non_max_suppression(boxes)
    assert set(filtered_boxes.scores) == {0.65, 0.7}


def test_nms_handles_empty_return():
    data = np.array([[0, 0, 1, 1, 0.4, 0]])
    boxes = FrameAugmentedBoxes(data, ptype=PType.absolute_tlwh)
    filtered_boxes = non_max_suppression(boxes)
    assert len(filtered_boxes) == 0


def test_nms_handles_empty_input():
    data = np.zeros((0, 6))
    boxes = FrameAugmentedBoxes(data, ptype=PType.absolute_tlwh)
    filtered_boxes = non_max_suppression(boxes)
    assert len(filtered_boxes) == 0
