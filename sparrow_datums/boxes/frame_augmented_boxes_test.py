import doctest

import numpy as np
import pytest

from sparrow_datums.boxes.frame_boxes import FrameBoxes

from ..chunk_types import PType
from ..exceptions import ValidationError
from . import frame_augmented_boxes
from .frame_augmented_boxes import FrameAugmentedBoxes
from .single_augmented_box import SingleAugmentedBox


def test_docstring_example():
    result = doctest.testmod(frame_augmented_boxes)
    assert result.attempted > 0
    assert result.failed == 0


def test_frame_augmented_boxes_conversion_creates_frame_augmented_boxes():
    boxes_a = FrameAugmentedBoxes(np.ones((2, 6)), ptype=PType.absolute_tlwh)
    boxes_b = boxes_a.to_tlbr()
    assert boxes_b.is_tlbr
    assert isinstance(boxes_b, FrameAugmentedBoxes)


def test_frame_augmented_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValidationError):
        FrameAugmentedBoxes(np.ones(6))
    with pytest.raises(ValidationError):
        FrameAugmentedBoxes(np.ones((2, 4)))


def test_frame_augmented_boxes_iterator_makes_frame_augmented_boxes():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), ptype=PType.absolute_tlwh)
    for box in boxes:
        assert isinstance(box, SingleAugmentedBox)
        assert box.ptype == PType.absolute_tlwh


def test_from_dict_with_empty_data():
    boxes = FrameAugmentedBoxes.from_dict(
        {
            "data": [],
            "ptype": "unknown",
            "image_width": None,
            "image_height": None,
            "fps": None,
            "object_ids": None,
        }
    )
    assert isinstance(boxes, FrameAugmentedBoxes)
    assert len(boxes) == 0


def test_from_single_box():
    box = SingleAugmentedBox(np.ones(6), PType.absolute_tlbr)
    frame_augmented_boxes = FrameAugmentedBoxes.from_single_box(box)
    assert frame_augmented_boxes.shape == (1, 6)
    assert frame_augmented_boxes.ptype == PType.absolute_tlbr


def test_add_box():
    box = SingleAugmentedBox(np.ones(6), PType.absolute_tlbr)
    frame_augmented_boxes = FrameAugmentedBoxes.from_single_box(box)
    frame_augmented_boxes = frame_augmented_boxes.add_box(box)
    assert frame_augmented_boxes.shape == (2, 6)


def test_get_single_box():
    boxes = FrameAugmentedBoxes(np.array([[1, 2, 3, 4, 0.9, 1]]), PType.absolute_tlwh)
    box = boxes.get_single_box(0)
    assert box.shape == (6,)
    assert box.label == 1


def test_to_frame_boxes():
    boxes = FrameAugmentedBoxes(np.array([[1, 2, 3, 4, 0.9, 1]]), PType.absolute_tlwh)
    boxes = boxes.to_frame_boxes()
    assert isinstance(boxes, FrameBoxes)
    assert boxes.ptype == PType.absolute_tlwh
    assert boxes.shape[-1] == 4
