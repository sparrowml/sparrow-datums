import numpy as np
import pytest

from sparrow_datums.boxes import (
    Boxes,
    BoxType,
    intersection_over_union,
    non_max_suppression,
)


def intersection_over_union_differnt_box_types_throw_value_error():
    boxes_a = Boxes(np.ones(4), BoxType.XYXY_ABS)
    boxes_b = Boxes(np.ones(4), BoxType.XYWH_ABS)
    with pytest.raises(ValueError):
        intersection_over_union(boxes_a, boxes_b)


def non_max_suppresion_does_nothing():
    boxes = Boxes(np.ones(4))
    non_max_suppression(boxes)
