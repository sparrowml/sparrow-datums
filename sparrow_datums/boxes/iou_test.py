import numpy as np

from ..types import BoxType
from .boxes import Boxes
from .frame_boxes import FrameAugmentedBoxes, FrameBoxes
from .iou import area, intersection, pairwise_iou
from .single_box import SingleBox


def test_intersection_calculation():
    box_a = SingleBox(np.ones(4), BoxType.absolute_tlwh)
    box_b = SingleBox(np.ones(4) * 2, BoxType.absolute_tlwh)
    assert intersection(box_a, box_a) == 1
    assert intersection(box_b, box_b) == 4
    assert intersection(box_a, box_b) == 0


def test_intersection_works_on_frame_boxes():
    boxes = FrameBoxes(np.ones((4, 4)), BoxType.absolute_tlwh)
    result = intersection(boxes, boxes)
    np.testing.assert_equal(result, np.ones(4))


def test_pairwise_intersection_works():
    boxes = Boxes(np.ones((5, 4)), BoxType.absolute_tlwh)
    result = intersection(boxes[:, None], boxes[None])
    assert result.shape == (5, 5)
    np.testing.assert_equal(result, 1)


def test_area_calculation():
    box = SingleBox(np.ones(4) * 3, BoxType.absolute_tlwh)
    assert area(box) == 9
    boxes = FrameBoxes(np.stack([np.arange(3)] * 4, -1), BoxType.absolute_tlwh)
    np.testing.assert_array_equal(area(boxes), np.array([0, 1, 4]))


def test_pairwise_iou_creates_pairwise_shape():
    boxes = FrameBoxes(np.ones((5, 4)), BoxType.absolute_tlwh)
    result = pairwise_iou(boxes, boxes)
    assert result.shape == (5, 5)


def test_pairwise_iou_works_for_frame_augmented_boxes():
    boxes = FrameAugmentedBoxes(np.ones((5, 6)), BoxType.absolute_tlwh)
    result = pairwise_iou(boxes, boxes)
    assert result.shape == (5, 5)
