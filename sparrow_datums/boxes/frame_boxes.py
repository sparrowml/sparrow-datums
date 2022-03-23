from typing import Iterator

import numpy as np
import numpy.typing as npt

from .boxes import Boxes
from .single_box import SingleBox


def _is_2d(x: npt.NDArray[np.float64]) -> None:
    if x.ndim != 2:
        raise ValueError("A frame boxes object must be a 2D array")


class FrameBoxes(Boxes):
    """A 2D frame of boxes."""

    def validate(self) -> None:
        """Check validity of boxes array."""
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleBox]:
        """Yield SingoeBox objects for each box."""
        for box in self.view(Boxes):
            yield box.view(SingleBox)
