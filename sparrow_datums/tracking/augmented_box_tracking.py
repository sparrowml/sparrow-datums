"""AugmentedBoxTracking chunk."""
from ..boxes import AugmentedBoxes


class AugmentedBoxTracking(AugmentedBoxes):
    """
    Dense data arrays for box tracking with [boxes, scores, labels].

    It inherits from :class:`.AugmentedBoxes`.
    The underlying NumPy array should have shape ``(n_frames, n_objects, 6)``.

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
