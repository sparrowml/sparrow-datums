"""Non-max suppression."""
import numpy as np

from .frame_augmented_boxes import FrameAugmentedBoxes


def non_max_suppression(
    boxes: FrameAugmentedBoxes,
    minimum_score: float = 0.5,
    iou_threshold: float = 0.5,
) -> FrameAugmentedBoxes:
    """Suppress overlapping boxes by box confidence and IoU."""
    keepers: list[int] = []
    score_mask = boxes.scores > minimum_score
    class_indices = sorted(set(boxes.labels.ravel()))
    for class_index in class_indices:
        mask = np.logical_and(score_mask, boxes.labels == class_index)
        mask_indices = np.argwhere(mask).ravel()
        box_candidates = boxes[mask]
        areas = (box_candidates.w + 1) * (box_candidates.h + 1)
        score_order = box_candidates.scores.argsort()[::-1]
        while score_order.size > 0:
            i = score_order[0]
            keepers.append(mask_indices[i])
            xx1 = np.maximum(box_candidates.x1[i], box_candidates.x1[score_order[1:]])
            yy1 = np.maximum(box_candidates.y1[i], box_candidates.y1[score_order[1:]])
            xx2 = np.minimum(box_candidates.x2[i], box_candidates.x2[score_order[1:]])
            yy2 = np.minimum(box_candidates.y2[i], box_candidates.y2[score_order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
            intersection = w * h
            iou = intersection / (areas[i] + areas[score_order[1:]] - intersection)

            index = np.where(iou <= iou_threshold)[0]
            score_order = score_order[index + 1]

    if len(keepers):
        return boxes[np.array(keepers)]
    return FrameAugmentedBoxes(
        np.zeros((0, 6)),
        ptype=boxes.ptype,
        **boxes.metadata_kwargs,
    )
