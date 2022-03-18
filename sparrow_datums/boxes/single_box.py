import numpy as np

from .augmented_boxes import AugmentedBoxes
from .boxes import Boxes


def _is_1d(x: np.ndarray) -> None:
    if x.ndim != 1:
        raise ValueError("Single box must be a 1D array")


class SingleBox(Boxes):
    def validate(self) -> None:
        super().validate()
        _is_1d(self)


class SingleAugmentedBox(AugmentedBoxes):
    def validate(self) -> None:
        super().validate()
        _is_1d(self)

    @property
    def score(self) -> float:
        return float(self.scores)

    @property
    def label(self) -> int:
        return int(self.labels)
