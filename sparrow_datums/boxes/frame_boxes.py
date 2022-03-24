from typing import Iterator

from ..types import FloatArray
from .boxes import Boxes
from .single_box import SingleBox


def _is_2d(x: FloatArray) -> None:
    if x.ndim != 2:
        raise ValueError("A frame boxes object must be a 2D array")


class FrameBoxes(Boxes):
    """
    2D dense data arrays for boxes.

    It inherits from :class:`.Boxes`.
    The underlying NumPy array should have shape ``(n_boxes, 4)``.

    Parameters
    ----------
    data : FloatArray
        A numpy array of dense floats
    ptype : PType
        The parameterization of the dense data
    image_width : int, optional
        The width of the relevant image
    image_height : int, optional
        The height of the relevant image

    Example
    -------
    >>> import numpy as np
    >>> from sparrow_datums import FrameBoxes, PType
    >>> boxes = FrameBoxes(np.array([[0, 0, 1, 1]]))
    >>> for box in boxes: print(box)
    [0 0 1 1]
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleBox]:
        """Yield SingoeBox objects for each box."""
        for box in self.view(Boxes):
            yield box.view(SingleBox)
