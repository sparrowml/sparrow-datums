from typing import TypeVar

import numpy as np

from ..chunk import Chunk
from ..exceptions import ValidationError

# from .keypoint_types import FloatArray, PType
from ..types import FloatArray, PType
from .keypoints import Keypoints

# from sparrow_datums import Keypoints


T = TypeVar("T", bound="Heatmaps")

# Note to self: np.view is making a shallow copy of an array.
class Heatmaps(Chunk):
    """Dense data arrays for keypoints.

    Parameters
    ----------
    Chunk : Chunk
        Base class for dense data arrays with metadata.

    Example
    -------
    # >>> import numpy as np
    # >>> from sparrow_datums import Keypoints, Heatmaps, PType
    # >>> Keypoints(np.ones(2))
    # Heatmaps([1., 1.])
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        if not self.shape or len(self.shape) > 3:
            raise ValidationError("Heatmaps cannot have more than 3 dimensions")
        if np.max(self) < 0 or np.min(self) > 1:
            raise ValidationError("Heatmap values has to be between 0 - 1")

    def validate_known_ptype(self) -> None:
        """Make sure PType is a known box parameterization."""
        known_box_parameterizations = {PType.heatmap}
        if self.ptype not in known_box_parameterizations:
            raise ValidationError(f"Unknown box parameterization: {self.ptype.name}")

    def to_keypoint(self, heatmaps):
        n_heatmaps = heatmaps.shape[0]
        keypoints = []
        for i in range(n_heatmaps):
            heatmap = heatmaps[i]
            ncols = heatmap.shape[-1]
            flattened_keypoint_indices = heatmap.flatten().argmax(-1)
            x = flattened_keypoint_indices % ncols
            y = np.floor(flattened_keypoint_indices / ncols)
            keypoint = np.array([x, y], dtype=float)
            keypoints.append(keypoint)
        return Keypoints(np.stack(keypoints)).to_absolute()
