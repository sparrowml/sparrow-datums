"""Intersection over union operations."""
from typing import Union, no_type_check

import numpy as np
import numpy.typing as npt

from .boxes import Boxes
from .frame_augmented_boxes import FrameAugmentedBoxes
from .frame_boxes import FrameBoxes


def intersection(boxes_a: Boxes, boxes_b: Boxes) -> npt.NDArray[np.float64]:
    """
    Intersection areas of two sets of boxes.

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
    # Define the inner box
    x1 = np.maximum(boxes_a.x1, boxes_b.x1)
    y1 = np.maximum(boxes_a.y1, boxes_b.y1)
    x2 = np.minimum(boxes_a.x2, boxes_b.x2)
    y2 = np.minimum(boxes_a.y2, boxes_b.y2)
    result: npt.NDArray[np.float64] = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    return result


def area(boxes: Boxes) -> npt.NDArray[np.float64]:
    """
    Areas of boxes.

    Parameters
    ----------
    boxes : Boxes
        A boxes array

    Returns
    -------
    areas : np.ndarray
        An ndarray of areas
    """
    return boxes.w * boxes.h


@no_type_check
def pairwise_iou(
    boxes_a: Union[FrameAugmentedBoxes, FrameBoxes],
    boxes_b: Union[FrameAugmentedBoxes, FrameBoxes],
) -> npt.NDArray[np.float64]:
    """
    Pairwise IoU for two sets of frame boxes.

    Parameters
    ----------
    boxes_a : FrameAugmentedBoxes or FrameBoxes
        A frame of boxes
    boxes_b : FrameAugmentedBoxes or FrameBoxes
        A frame of boxes

    Returns
    -------
    iou_scores : np.ndarray
        A (n_boxes_a, n_boxes_b) matrix of iou scores
    """
    boxes_a_parent = boxes_a.__class__.__base__
    boxes_a_view: Union[FrameAugmentedBoxes, FrameBoxes] = boxes_a.view(boxes_a_parent)
    boxes_a_expanded = boxes_a_view[:, None, :]
    boxes_b_parent = boxes_b.__class__.__base__
    boxes_b_view: Union[FrameAugmentedBoxes, FrameBoxes] = boxes_b.view(boxes_b_parent)
    boxes_b_expanded = boxes_b_view[None, :, :]
    intersections = intersection(boxes_a_expanded, boxes_b_expanded)
    areas = area(boxes_a_expanded) + area(boxes_b_expanded)
    unions = areas - intersections
    unions[unions == 0] = np.inf
    return intersections / unions
