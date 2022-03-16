import numpy as np
import pytest

from sparrow_datums.boxes import Boxes, FrameBoxes, SingleBox
from sparrow_datums.types import BoxType


def test_bad_shape_throws_type_error():
    with pytest.raises(ValueError):
        Boxes(np.ones(3))


def test_boxes_to_relative_moves_boxes_to_0_1():
    image_size = 512
    boxes_a = Boxes(
        np.random.uniform(size=(10, 4)) * image_size,
        BoxType.absolute_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_relative()
    assert boxes_b.is_relative
    np.testing.assert_array_less(boxes_b.array, 1.0)
    boxes_c = boxes_b.to_relative()
    assert boxes_c.is_relative


def test_boxes_to_absolute_moves_boxes_to_0_image_size():
    image_size = 512
    boxes_a = Boxes(
        np.random.uniform(size=(10, 4)),
        BoxType.relative_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_absolute()
    assert boxes_b.is_absolute
    assert (boxes_b.array < 1).mean() < 0.1
    boxes_c = boxes_b.to_absolute()
    assert boxes_c.is_absolute


def test_boxes_to_tlbr_moves_boxes_to_tlbr():
    x = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = Boxes(x, BoxType.relative_tlwh).to_tlbr()
    np.testing.assert_equal(boxes.array, 1)


def test_boxes_to_tlwh_moves_boxes_to_tlwh():
    result = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = Boxes(np.ones((5, 4)), BoxType.relative_tlbr).to_tlwh()
    np.testing.assert_equal(boxes.array, result)


def test_boxes_to_tlbr_doesnt_change_original_instance():
    x = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = Boxes(x, BoxType.relative_tlwh)
    _ = boxes.to_tlbr()
    assert boxes.is_tlwh
    assert boxes.is_relative


def test_box_slicing_cols_throws_value_error():
    boxes = Boxes(np.random.uniform(size=(10, 4)))
    with pytest.raises(ValueError):
        top_left = boxes[:, :2]
    top_left = boxes.array[:, :2]
    assert isinstance(top_left, np.ndarray)


def test_boxes_deserializes_type():
    boxes = Boxes.from_dict(
        {
            "data": [1, 2, 3, 4],
            "type": "relative_tlbr",
            "image_width": None,
            "image_height": None,
            "fps": None,
            "object_ids": None,
        }
    )
    assert boxes.type


def test_boxes_saves_classname():
    data = Boxes(np.ones(4)).to_dict()
    assert data["classname"] == "Boxes"


def test_single_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleBox(np.ones((2, 4)))
    with pytest.raises(ValueError):
        SingleBox(np.ones(3))
    # This should work
    SingleBox(np.random.uniform(size=4))


def test_frame_boxes_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        FrameBoxes(np.ones(4))
    with pytest.raises(ValueError):
        FrameBoxes(np.ones((2, 3)))
    # This should work
    FrameBoxes(np.random.uniform(size=(2, 4)))


def test_frame_boxes_creates_more_frame_boxes():
    boxes_a = FrameBoxes(np.ones((2, 4)), BoxType.relative_tlbr)
    boxes_b = boxes_a.to_tlwh()
    assert isinstance(boxes_b, FrameBoxes)
