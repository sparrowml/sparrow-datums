from ..types import FloatArray
from .boxes import Boxes


def _is_1d(x: FloatArray) -> None:
    if x.ndim != 1:
        raise ValueError("Single box must be a 1D array")


class SingleBox(Boxes):
    """
    1D dense data arrays for a box.

    It inherits from :class:`.Boxes`.
    The underlying NumPy array should have shape ``(4,)``.

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
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        super().validate()
        _is_1d(self)
