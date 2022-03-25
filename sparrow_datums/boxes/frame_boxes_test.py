import doctest

import numpy as np
import pytest

from ..types import PType
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
