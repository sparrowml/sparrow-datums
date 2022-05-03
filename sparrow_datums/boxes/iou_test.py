import numpy as np

from ..types import PType
from .boxes import Boxes
from .frame_augmented_boxes import FrameAugmentedBoxes
from .frame_boxes import FrameBoxes
from .iou import area, intersection, pairwise_iou
from .single_box import SingleBox


def test_intersection_calculation():
    box_a = SingleBox(np.ones(4), PType.absolute_tlwh)
    box_b = SingleBox(np.ones(4) * 2, PType.absolute_tlwh)
    assert intersection(box_a, box_a) == 1
    assert intersection(box_b, box_b) == 4
    assert intersection(box_a, box_b) == 0


def test_intersection_works_on_frame_boxes():
    boxes = FrameBoxes(np.ones((4, 4)), PType.absolute_tlwh)
    result = intersection(boxes, boxes)
    np.testing.assert_equal(result, np.ones(4))


def test_pairwise_intersection_works():
    boxes = Boxes(np.ones((5, 4)), PType.absolute_tlwh)
    result = intersection(boxes[:, None], boxes[None])
    assert result.shape == (5, 5)
    np.testing.assert_equal(result, 1)


def test_area_calculation():
    box = SingleBox(np.ones(4) * 3, PType.absolute_tlwh)
    assert area(box) == 9
    boxes = FrameBoxes(np.stack([np.arange(3)] * 4, -1), PType.absolute_tlwh)
    np.testing.assert_array_equal(area(boxes), np.array([0, 1, 4]))


def test_pairwise_iou_creates_pairwise_shape():
    boxes = FrameBoxes(np.ones((5, 4)), PType.absolute_tlwh)
    result = pairwise_iou(boxes, boxes)
    assert result.shape == (5, 5)


def test_pairwise_iou_works_for_frame_augmented_boxes():
    boxes = FrameAugmentedBoxes(np.ones((5, 6)), PType.absolute_tlwh)
    result = pairwise_iou(boxes, boxes)
    assert result.shape == (5, 5)


def test_max_iou_is_1():
    frame_a = FrameAugmentedBoxes(
        [[130.02, 582.8, 87.96, 57.14, 1.0, 2.0]],
        PType.absolute_tlwh,
        image_width=1920,
        image_height=1080,
    )
    frame_b = FrameAugmentedBoxes(
        [[1.82, 499.01, 41.07, 38.85, 1.0, 3.0]],
        PType.absolute_tlwh,
        image_width=1920,
        image_height=1080,
    )
    assert pairwise_iou(frame_a, frame_b) < 1
