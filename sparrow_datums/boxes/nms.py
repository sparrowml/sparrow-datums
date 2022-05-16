import numpy as np

from .augmented_boxes import AugmentedBoxes
from .iou import area


def non_max_suppression(boxes: AugmentedBoxes, iou_threshold: float) -> AugmentedBoxes:
    """
    Remove overlapping boxes with non-max suppression.

    Parameters
    ----------
    boxes
        Augmented box object with scores
    iou_threshold
        The IoU threshold above which to remove overlapping boxes

    Returns
    -------
    filtered_boxes
        A set of boxes to keep
    """
    areas = area(boxes)
    order = boxes.scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes.x1[i], boxes.x1[order[1:]])
        yy1 = np.maximum(boxes.y1[i], boxes.y1[order[1:]])
        xx2 = np.minimum(boxes.x2[i], boxes.x2[order[1:]])
        yy2 = np.minimum(boxes.y2[i], boxes.y2[order[1:]])

        max_width = np.maximum(0.0, xx2 - xx1 + 1)
        max_height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = max_width * max_height
        ious = intersection / (areas[i] + areas[order[1:]] - intersection)
        indexes = np.where(ious <= iou_threshold)[0]
        order = order[indexes + 1]
    filtered_boxes: AugmentedBoxes = boxes[np.array(keep)]
    return filtered_boxes
