from .boxes import Boxes


def intersection_over_union(boxes_a: Boxes, boxes_b: Boxes) -> None:
    if boxes_a.box_type != boxes_b.box_type:
        raise ValueError(
            "All boxes must have the same type. "
            f"Found {boxes_a.box_type.name} and {boxes_b.box_type.name}."
        )


def non_max_suppression(boxes: Boxes) -> None:
    pass
