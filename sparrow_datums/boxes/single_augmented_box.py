from .augmented_boxes import AugmentedBoxes
from .single_box import _is_1d


class SingleAugmentedBox(AugmentedBoxes):
    def validate(self) -> None:
        super().validate()
        _is_1d(self)

    @property
    def score(self) -> float:
        return float(self.scores)

    @property
    def label(self) -> int:
        return int(self.labels)
