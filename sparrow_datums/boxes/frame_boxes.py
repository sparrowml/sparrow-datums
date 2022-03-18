from typing import Iterator

import numpy as np

from .augmented_boxes import AugmentedBoxes
from .boxes import Boxes
from .single_box import SingleAugmentedBox, SingleBox


def _is_2d(x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError("A frame boxes object must be a 2D array")


class FrameBoxes(Boxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleBox]:
        for box in self.view(Boxes):
            yield box.view(SingleBox)


class FrameAugmentedBoxes(AugmentedBoxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleAugmentedBox]:
        for box in self.view(AugmentedBoxes):
            yield box.view(SingleAugmentedBox)
