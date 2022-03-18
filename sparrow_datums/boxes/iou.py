import numpy as np

from .boxes import Boxes
from .frame_boxes import FrameBoxes


def intersection(boxes_a: Boxes, boxes_b: Boxes) -> np.ndarray:
    """
    Intersection areas of two sets of boxes

    Parameters
    ----------
    boxes_a : Boxes
        A boxes array
    boxes_b : Boxes
        A boxes array

    Returns
    -------
    intersections : np.ndarray
        An ndarray of intersection areas
    """
    a = boxes_a.to_tlbr().array
    b = boxes_b.to_tlbr().array
    # Define the inner box
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    widths = np.absolute(x2 - x1)
    heights = np.absolute(y2 - y1)
    return np.maximum(widths, 0) * np.maximum(heights, 0)


def area(boxes: Boxes) -> np.ndarray:
    """
    Areas of boxes

    Parameters
    ----------
    boxes : Boxes
        A boxes array

    Returns
    -------
    areas : np.ndarray
        An ndarray of areas
    """
    x = boxes.to_tlbr().array
    widths = np.absolute(x[..., 2] - x[..., 0])
    heights = np.absolute(x[..., 3] - x[..., 1])
    return widths * heights


def pairwise_iou(boxes_a: FrameBoxes, boxes_b: FrameBoxes) -> np.ndarray:
    """
    Pairwise IoU for two sets of frame boxes
    """
    boxes_a = boxes_a.to_tlbr().view(Boxes)[:, None, :]
    boxes_b = boxes_b.to_tlbr().view(Boxes)[None, :, :]
    intersections = intersection(boxes_a, boxes_b)
    areas = area(boxes_a) + area(boxes_b)
    unions = areas - intersections
    unions[unions == 0] = np.inf
    return intersections / unions
