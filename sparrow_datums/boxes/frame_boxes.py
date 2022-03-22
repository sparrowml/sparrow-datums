from typing import Iterator

import numpy as np
import numpy.typing as npt

from .boxes import Boxes
from .single_box import SingleBox


def _is_2d(x: npt.NDArray[np.float64]) -> None:
    if x.ndim != 2:
        raise ValueError("A frame boxes object must be a 2D array")


class FrameBoxes(Boxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleBox]:
        for box in self.view(Boxes):
            yield box.view(SingleBox)
