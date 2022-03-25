from .augmented_boxes import AugmentedBoxes
from .single_box import SingleBox, _is_1d


class SingleAugmentedBox(AugmentedBoxes):
    """
    1D dense data arrays for an augmented box.

    The data contain ``[box, score, label]`` components.
    It inherits from :class:`.AugmentedBoxes`.
    The underlying NumPy array should have shape ``(6,)``.

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

    @property
    def score(self) -> float:
        """Singular alias for scores."""
        return float(self.scores)

    @property
    def label(self) -> int:
        """Singular alias for labels."""
        return int(self.labels)

    def to_single_box(self) -> SingleBox:
        """Convert to SingleBox."""
        return SingleBox(
            self.array[:4],
            ptype=self.ptype,
            **self.metadata_kwargs,
        )
