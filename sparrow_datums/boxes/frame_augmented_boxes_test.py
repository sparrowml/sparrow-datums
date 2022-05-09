import doctest
import os
import tempfile

import numpy as np
import pytest

from sparrow_datums.boxes.frame_boxes import FrameBoxes

from ..types import PType
from . import frame_augmented_boxes
from .frame_augmented_boxes import FrameAugmentedBoxes
from .single_augmented_box import SingleAugmentedBox

DARWIN_DICT = {
    "dataset": "RetinaNet Detections",
    "image": {
        "width": 1920,
        "height": 1080,
        "original_filename": "Ring_541700402_1820_7075017446887328011_00000.jpg",
        "filename": "Ring_541700402_1820_7075017446887328011_00000.jpg",
        "url": "https://darwin.v7labs.com/api/images/335989560/original",
        "thumbnail_url": "https://darwin.v7labs.com/api/images/335989560/thumbnail",
        "path": "/",
        "workview_url": "https://darwin.v7labs.com/workview?dataset=417732&image=4296",
    },
    "annotations": [
        {
            "bounding_box": {"h": 301.37, "w": 219.31, "x": 756.71, "y": 686.49},
            "name": "bicycle",
        },
        {
            "bounding_box": {"h": 480.11, "w": 527.55, "x": 1010.4, "y": 506.5},
            "name": "car",
        },
        {
            "bounding_box": {"h": 104.65, "w": 80.25, "x": 243.55, "y": 592.7},
            "name": "chair",
        },
        {
            "bounding_box": {"h": 30.25, "w": 59.21, "x": 704.58, "y": 418.17},
            "name": "car",
        },
        {
            "bounding_box": {"h": 46.23, "w": 74.03, "x": 1796.27, "y": 482.56},
            "name": "car",
        },
        {
            "bounding_box": {"h": 57.3, "w": 89.47, "x": 129.04, "y": 582.63},
            "name": "bicycle",
        },
        {
            "bounding_box": {"h": 16.48, "w": 34.51, "x": 658.83, "y": 397.2},
            "name": "car",
        },
    ],
}


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
    with pytest.raises(ValueError):
        FrameAugmentedBoxes(np.ones(6))
    with pytest.raises(ValueError):
        FrameAugmentedBoxes(np.ones((2, 4)))


def test_frame_augmented_boxes_iterator_makes_frame_augmented_boxes():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), ptype=PType.absolute_tlwh)
    for box in boxes:
        assert isinstance(box, SingleAugmentedBox)
        assert box.ptype == PType.absolute_tlwh


def test_frame_augmented_boxes_can_become_darwin_dict():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), ptype=PType.absolute_tlwh)
    data = boxes.to_darwin_dict("foobar")
    assert data["image"]["filename"] == "foobar"
    for annotation in data["annotations"]:
        assert "bounding_box" in annotation
        assert annotation["name"] == "Unknown"


def test_darwin_dict_is_json_serializable():
    boxes = FrameAugmentedBoxes(np.ones((2, 6)), ptype=PType.absolute_tlwh)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json")
        boxes.to_darwin_file(path, "foobar")


def test_from_darwin_dict_creates_frame_augmented_boxes():
    boxes = FrameAugmentedBoxes.from_darwin_dict(DARWIN_DICT)
    assert len(boxes) == 7
    assert isinstance(boxes, FrameAugmentedBoxes)
    assert boxes.ptype == PType.absolute_tlwh
    assert boxes.image_width
    assert boxes.image_height
    np.testing.assert_equal(boxes.labels, -1)


def test_from_darwin_dict_creates_frame_augmented_boxes_with_no_boxes():
    empty_darwin_dict = DARWIN_DICT.copy()
    empty_darwin_dict["annotations"] = []
    boxes = FrameAugmentedBoxes.from_darwin_dict(empty_darwin_dict)
    assert isinstance(boxes, FrameAugmentedBoxes)


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


def test_from_darwin_dict_creates_frame_augmented_boxes_with_non_bounding_box_annotations():
    non_box_dict = DARWIN_DICT.copy()
    non_box_dict["annotations"] = [{"foo": "bar"}]
    boxes = FrameAugmentedBoxes.from_darwin_dict(non_box_dict)
    assert isinstance(boxes, FrameAugmentedBoxes)


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
    labels = ["car", "bicycle", "chair"]
    boxes = FrameAugmentedBoxes.from_darwin_dict(DARWIN_DICT, label_names=labels)
    box = boxes.get_single_box(-1)
    assert box.shape == (6,)
    assert labels[box.label] == "car"


def test_to_frame_boxes():
    boxes = FrameAugmentedBoxes.from_darwin_dict(DARWIN_DICT)
    boxes = boxes.to_frame_boxes()
    assert isinstance(boxes, FrameBoxes)
    assert boxes.ptype == PType.absolute_tlwh
    assert boxes.shape[-1] == 4
