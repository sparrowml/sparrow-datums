from typing import TypeVar

import numpy as np

from sparrow_datums.keypoints.keypoints import Keypoints

from ..chunk import Chunk
from ..exceptions import ValidationError
from ..types import FloatArray, PType
from .keypoints import Keypoints

T = TypeVar("T", bound="Heatmaps")


class Heatmaps(Chunk):
    """Dense data arrays for keypoints.

    Parameters
    ----------
    Chunk : Chunk
        Base class for dense data arrays with metadata.

    Example
    -------
    >>> import numpy as np
    >>> from sparrow_datums import Keypoints, Heatmaps, PType
    >>> n_keypoints = 2
    >>> np.random.seed(0)
    >>> candidate_arr = np.random.randint(low=2, high=20, size=(n_keypoints, 2))
    >>> keypoint = Keypoints(candidate_arr,PType.absolute_kp,image_height=21,image_width=28)
    >>> heatmap_array = keypoint.to_heatmap_array()
    >>> heatmap = Heatmaps(heatmap_array,PType.heatmap,image_width=heatmap_array.shape[-1],image_height=heatmap_array.shape[-2])
    >>> back_to_keypoint = heatmap.to_keypoint()
    >>> back_to_keypoint
    Keypoints([[14., 17.],
               [ 2.,  5.]])
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
        keypoints = []
        heatmap_dims = len(heatmaps.shape)
        if heatmap_dims == 2:
            heatmaps = np.expand_dims(heatmaps, axis=2)
            heatmaps = np.moveaxis(heatmaps, -1, 0)
        n_heatmaps = heatmaps.shape[0]
        for i in range(n_heatmaps):
            heatmap = heatmaps[i, :, :]
            _, width = heatmap.shape
            ncols = width
            flattened_keypoint_indices = heatmap.flatten().argmax(-1)
            x = flattened_keypoint_indices % ncols
            y = np.floor(flattened_keypoint_indices / ncols)
            keypoint = np.array([x, y], dtype=float)
            keypoints.append(keypoint)
        return Keypoints(np.stack(keypoints), **self.metadata_kwargs)
