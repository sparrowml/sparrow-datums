import os
import tempfile

import numpy as np
import pytest

from ..types import BoxType
from .frame_boxes import FrameAugmentedBoxes, FrameBoxes
from .single_box import SingleAugmentedBox, SingleBox


def test_frame_boxes_conversion_creates_frame_boxes():
    boxes_a = FrameBoxes(np.ones((2, 4)), BoxType.relative_tlbr)
    boxes_b = boxes_a.to_tlwh()
    assert isinstance(boxes_b, FrameBoxes)


def test_frame_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        FrameBoxes(np.ones(4))
    with pytest.raises(ValueError):
        FrameBoxes(np.ones((2, 3)))


def test_frame_boxes_iterator_makes_single_boxes():
    boxes = FrameBoxes(np.ones((2, 4)), BoxType.relative_tlbr)
    for box in boxes:
        assert isinstance(box, SingleBox)
        assert box.type == BoxType.relative_tlbr


def test_frame_augmented_boxes_conversion_creates_frame_augmented_boxes():
    boxes_a = FrameAugmentedBoxes(np.ones((2, 6)), type=BoxType.absolute_tlwh)
    boxes_b = boxes_a.to_tlbr()
    assert boxes_b.is_tlbr
    assert isinstance(boxes_b, FrameAugmentedBoxes)


def test_frame_augmented_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        FrameAugmentedBoxes(np.ones(6))
    with pytest.raises(ValueError):
        FrameAugmentedBoxes(np.ones((2, 4)))


def test_frame_augmented_boxes_iterator_makes_frame_augmented_boxes():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), type=BoxType.absolute_tlwh)
    for box in boxes:
        assert isinstance(box, SingleAugmentedBox)
        assert box.type == BoxType.absolute_tlwh


def test_frame_augmented_boxes_can_become_darwin_dict():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), type=BoxType.absolute_tlwh)
    data = boxes.to_darwin_dict("foobar")
    assert data["image"]["filename"] == "foobar"
    for annotation in data["annotations"]:
        assert "bounding_box" in annotation
        assert annotation["name"] == "Unknown"


def test_darwin_dict_is_json_serializable():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), type=BoxType.absolute_tlwh)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json")
        boxes.to_darwin_annotation_file(path, "foobar")
