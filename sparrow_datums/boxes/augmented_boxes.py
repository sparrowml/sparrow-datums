from typing import List

import numpy as np

from .boxes import Boxes


class AugmentedBoxes(Boxes):
    """A (..., 6) boxes array with [boxes, scores, labels] components"""

    def validate(self) -> None:
        if not self.shape or self.shape[-1] != 6:
            raise ValueError("AugmentedBoxes arrays must have size-6 dimensions")
        if not np.all(np.mod(self.array[..., -1], 1) == 0):
            raise ValueError("labels must be whole number class indices")
        if not np.all(self.scores >= 0) or not np.all(self.scores <= 1):
            raise ValueError("scores array must be floats in [0, 1]")

    @property
    def scores(self) -> np.ndarray:
        return self.array[..., -2]

    @property
    def labels(self) -> np.ndarray:
        return self.array[..., -1].astype(int)

    def names(self, label_names: List[str]) -> np.ndarray:
        return np.array(label_names)[self.labels]
