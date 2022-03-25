import doctest

import numpy as np
import pytest

from ..types import FloatArray, PType
from . import frame_boxes
from .frame_boxes import FrameBoxes
from .single_box import SingleBox


def test_docstring_example():
    result = doctest.testmod(frame_boxes)
    assert result.attempted > 0
    assert result.failed == 0


def test_frame_boxes_conversion_creates_frame_boxes():
    boxes_a = FrameBoxes(np.ones((2, 4)), PType.relative_tlbr)
    boxes_b = boxes_a.to_tlwh()
    assert isinstance(boxes_b, FrameBoxes)


def test_frame_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        FrameBoxes(np.ones(4))
    with pytest.raises(ValueError):
        FrameBoxes(np.ones((2, 3)))


def test_frame_boxes_iterator_makes_single_boxes():
    boxes = FrameBoxes(np.ones((2, 4)), PType.relative_tlbr)
    for box in boxes:
        assert isinstance(box, SingleBox)
        assert box.ptype == PType.relative_tlbr


def test_from_single_box_preserves_data():
    box = SingleBox(np.random.uniform(size=4))
    frame_boxes = FrameBoxes.from_single_box(box)
    np.testing.assert_equal(box.array, frame_boxes.array.ravel())
    assert box.ptype == frame_boxes.ptype


def test_add_box_with_different_attributes_fails():
    data: FloatArray = np.random.uniform(size=4)
    boxes = FrameBoxes.from_single_box(SingleBox(data))
    box_b = SingleBox(data, ptype=PType.relative_tlbr)
    with pytest.raises(ValueError):
        boxes.add_box(box_b)
    box_c = SingleBox(data, image_width=100)
    with pytest.raises(ValueError):
        boxes.add_box(box_c)


def test_add_box_with_same_attributes_works():
    data: FloatArray = np.random.uniform(size=4)
    boxes = FrameBoxes.from_single_boxes([SingleBox(data)])
    box_b = SingleBox(data + 1)
    new_boxes = boxes.add_box(box_b)
    assert len(new_boxes) == 2


def test_get_single_box_returns_single_box():
    boxes = FrameBoxes(np.ones((2, 4)), PType.relative_tlbr)
    single_box = boxes.get_single_box(0)
    assert isinstance(single_box, SingleBox)
    assert single_box.ptype == boxes.ptype
