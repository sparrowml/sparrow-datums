from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..types import FloatArray
from .boxes import Boxes


class AugmentedBoxes(Boxes):
    """
    Dense data arrays for augmented boxes.

    The data contain ``[boxes, scores, labels]`` components.
    It inherits from :class:`.Boxes`.
    The underlying NumPy array should have shape ``(..., 6)``,
    with 4 dimensions reserved for boxes, and the last two for
    scores and labels respectively.

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
    >>> from sparrow_datums import AugmentedBoxes, PType
    >>> augmented_boxes = AugmentedBoxes(np.array([[0, 0, 1, 1, 0.5, 3]]))
    >>> augmented_boxes.scores
    array([0.5])
    >>> augmented_boxes.labels
    array([3])
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        if not self.shape or self.shape[-1] != 6:
            raise ValueError("AugmentedBoxes arrays must have size-6 dimensions")
        if not np.all(np.nan_to_num(np.mod(self.array[..., -1], 1)) == 0):
            raise ValueError("labels must be whole number class indices")
        nonan_scores = np.nan_to_num(self.scores)
        if not np.all(nonan_scores >= 0) or not np.all(nonan_scores <= 1):
            raise ValueError("scores array must be floats in [0, 1]")

    @property
    def scores(self) -> FloatArray:
        """Confidence scores."""
        result: npt.NDArray[np.float64] = self.array[..., -2]
        return result

    @property
    def labels(self) -> npt.NDArray[np.int64]:
        """Class label indices."""
        result: npt.NDArray[np.int64] = np.nan_to_num(
            self.array[..., -1], nan=-1
        ).astype(np.int64)
        return result

    def names(self, label_names: list[str]) -> npt.NDArray[np.str_]:
        """Map class label indices to string names."""
        result: npt.NDArray[np.str_] = np.array(label_names)[self.labels].astype(
            np.str_
        )
        return result
