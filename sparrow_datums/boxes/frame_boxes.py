import numpy as np

from .augmented_boxes import AugmentedBoxes
from .boxes import Boxes


def _is_2d(x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError("A frame boxes object must be a 2D array")


class FrameBoxes(Boxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)


class FrameAugmentedBoxes(AugmentedBoxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)
