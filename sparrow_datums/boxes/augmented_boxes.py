import numpy as np

from .boxes import Boxes


class AugmentedBoxes(Boxes):
    """A (..., 6) boxes array with [boxes, labels, scores] components"""

    def validate(self) -> None:
        if not self.shape or self.shape[-1] != 6:
            raise ValueError("AugmentedBoxes arrays must have size-6 dimensions")
        labels = self.array[..., -2]
        if not np.all(np.mod(labels, 1) == 0):
            raise ValueError("labels must be whole number class indices")
        scores = self.array[..., -1]
        if not np.all(scores >= 0) or not np.all(scores <= 1):
            raise ValueError("scores array must be floats in [0, 1]")
