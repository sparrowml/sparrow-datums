import numpy as np

from sparrow_datums import Boxes, BoxType
from sparrow_datums.iou import pairwise_iou


def test_pairwise_iou_on_tlwh_copies():
    x = np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], -1)
    boxes = Boxes(x, BoxType.relative_tlwh)
    _ = pairwise_iou(boxes, boxes)
    assert boxes.is_tlwh
    np.testing.assert_equal(boxes.array, x)
