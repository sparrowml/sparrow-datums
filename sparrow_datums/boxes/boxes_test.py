import numpy as np
import numpy.typing as npt
import pytest

from ..types import PType
from .boxes import Boxes


def test_bad_shape_throws_type_error():
    with pytest.raises(ValueError):
        Boxes(np.ones(3))


def test_to_relative_moves_boxes_to_0_1():
    image_size = 512
    boxes_a = Boxes(
        np.random.uniform(size=(10, 4)) * image_size,
        PType.absolute_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_relative()
    assert boxes_b.is_relative
    np.testing.assert_array_less(boxes_b.array, 1.0)
    boxes_c = boxes_b.to_relative()
    assert boxes_c.is_relative


def test_to_absolute_moves_boxes_to_0_image_size():
    image_size = 512
    boxes_a = Boxes(
        np.random.uniform(size=(10, 4)),
        PType.relative_tlwh,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_absolute()
    assert boxes_b.is_absolute
    assert (boxes_b.array < 1).mean() < 0.1
    boxes_c = boxes_b.to_absolute()
    assert boxes_c.is_absolute


def test_to_tlbr_moves_boxes_to_tlbr():
    x: npt.NDArray[np.float64] = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = Boxes(x, PType.relative_tlwh).to_tlbr()
    np.testing.assert_equal(boxes.array, 1)


def test_to_tlwh_moves_boxes_to_tlwh():
    result: npt.NDArray[np.float64] = np.concatenate(
        [np.ones((5, 2)), np.zeros((5, 2))], -1
    )
    boxes = Boxes(np.ones((5, 4)), PType.relative_tlbr).to_tlwh()
    np.testing.assert_equal(boxes.array, result)


def test_boxes_to_tlbr_doesnt_change_original_instance():
    x: npt.NDArray[np.float64] = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = Boxes(x, PType.relative_tlwh)
    _ = boxes.to_tlbr()
    assert boxes.is_tlwh
    assert boxes.is_relative


def test_box_slicing_cols_throws_value_error():
    boxes = Boxes(np.random.uniform(size=(10, 4)))
    with pytest.raises(ValueError):
        boxes[:, :2]
    top_left: npt.NDArray[np.float64] = boxes.array[:, :2]
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


def test_negative_box_values_throw():
    with pytest.raises(ValueError):
        Boxes(-np.ones(4))
