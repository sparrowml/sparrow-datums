"""BoxTracking chunk."""
from ..boxes import Boxes


class BoxTracking(Boxes):
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
        """Check shape of box tracking array."""
        if self.ndim != 3:
            raise ValueError("Tracking chunks must have 3 dimensions")
        super().validate()
