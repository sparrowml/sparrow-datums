import numpy as np
import numpy.typing as npt

from .boxes import Boxes


def _is_1d(x: npt.NDArray[np.float64]) -> None:
    if x.ndim != 1:
        raise ValueError("Single box must be a 1D array")


class SingleBox(Boxes):
    def validate(self) -> None:
        super().validate()
        _is_1d(self)
