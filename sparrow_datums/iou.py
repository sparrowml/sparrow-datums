import numpy as np

from sparrow_datums import Boxes


def pairwise_iou(boxes_a: Boxes, boxes_b: Boxes) -> np.ndarray:
    """
    Intersection over Union for two sets of boxes

    Parameters
    ----------
    boxes_a : Boxes
        A 2D array of boxes
    boxes_b : Boxes
        A 2D array of boxes

    Returns
    -------
    overlaps : np.ndarray
        A (n_boxes_a, n_boxes_b) matrix of IoU scores
    """
    # TODO: put this constraint on the boxes class
    assert boxes_a.ndim == 2 and boxes_b.ndim == 2
    boxes_a = boxes_a.to_tlbr()
    boxes_b = boxes_b.to_tlbr()
    return boxes_a.array.mean()
    # # Define the inner box
    # x1 = max(a[0], b[0])
    # y1 = max(a[1], b[1])
    # x2 = min(a[2], b[2])
    # y2 = min(a[3], b[3])
    # # Area of a, b separately
    # a_area = (a[2] - a[0]) * (a[3] - a[1])
    # b_area = (b[2] - b[0]) * (b[3] - b[1])
    # total_area = a_area + b_area
    # # Area of inner box
    # intersection = max(0, x2 - x1) * max(y2 - y1, 0)
    # # Area of union
    # union = total_area - intersection
    # return intersection / union
