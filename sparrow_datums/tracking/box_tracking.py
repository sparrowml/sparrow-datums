"""BoxTracking chunk."""
from typing import Iterator

from ..boxes import Boxes, FrameBoxes
from .tracking import Tracking


class BoxTracking(Tracking, Boxes):
    """
    Dense data arrays for box tracking.

    It inherits from :class:`.Boxes`.
    The underlying NumPy array should have shape ``(n_frames, n_objects, 4)``.

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
    fps : float, optional
        The framerate of the chunk data (if tracking)
    object_ids:  list[str], optional
        Identifiers for the objects (if tracking)
    """

    def validate(self) -> None:
        """Validate tracking shape and boxes."""
        super().validate()
        Boxes.validate(self)

    def __iter__(self) -> Iterator[FrameBoxes]:
        """Yield FrameBoxes objects for each frame."""
        for box in self.view(Boxes):
            yield box.view(FrameBoxes)
