import pytest

import numpy as np

from sparrow_datums.boxes import Boxes, BoxType


def test_bad_shape_throws_type_error():
    with pytest.raises(TypeError):
        Boxes(np.ones(3), BoxType.XYWH_REL)


def test_boxes_with_no_width_throws_on_scale():
    boxes = Boxes(np.ones(4), BoxType.XYWH_REL)
    with pytest.raises(ValueError):
        boxes.scale


def test_boxes_to_relative_moves_boxes_to_0_1():
    image_size = 512
    boxes_a = Boxes(
        np.random.uniform(size=(10, 4)) * image_size,
        BoxType.XYWH_ABS,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_relative()
    assert boxes_b.is_relative
    np.testing.assert_array_less(boxes_b, 1.0)
    boxes_c = boxes_b.to_relative()
    assert boxes_c.is_relative


def test_boxes_to_absolute_moves_boxes_to_0_image_size():
    image_size = 512
    boxes_a = Boxes(
        np.random.uniform(size=(10, 4)),
        BoxType.XYWH_REL,
        image_width=image_size,
        image_height=image_size,
    )
    boxes_b = boxes_a.to_absolute()
    assert boxes_b.is_absolute
    assert (boxes_b < 1).mean() < 0.01
    boxes_c = boxes_b.to_absolute()
    assert boxes_c.is_absolute
