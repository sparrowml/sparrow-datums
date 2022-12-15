from typing import TypeVar

import numpy as np

from sparrow_datums.keypoints.keypoints import Keypoints

from ..chunk import Chunk
from ..exceptions import ValidationError

# from .keypoint_types import FloatArray, PType
from ..types import FloatArray, PType
from .keypoints import Keypoints

# from sparrow_datums import Keypoints


T = TypeVar("T", bound="Heatmaps")


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

    @property
    def is_heatmap(self) -> bool:
        """Keypoint is in heatmap form."""
        return bool(self.ptype.is_heatmap)

    def to_keypoint(self):
        heatmaps = self.array
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
