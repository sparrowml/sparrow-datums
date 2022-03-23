from .augmented_boxes import AugmentedBoxes
from .single_box import _is_1d


class SingleAugmentedBox(AugmentedBoxes):
    """A single box with score and label."""

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
