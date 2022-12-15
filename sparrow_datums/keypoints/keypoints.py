from typing import TypeVar

import numpy as np

from ..chunk import Chunk
from ..exceptions import ValidationError
from ..types import FloatArray, PType

T = TypeVar("T", bound="Keypoints")

# Note to self: np.view is making a shallow copy of an array.
class Keypoints(Chunk):
    """Dense data arrays for keypoints.

    Parameters
    ----------
    Chunk : Chunk
        Base class for dense data arrays with metadata.

    Example
    -------
    >>> import numpy as np
    >>> from sparrow_datums import Keypoints, PType
    >>> Keypoints(np.ones(2))
    Keypoints([1., 1.])
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        if not self.shape or self.shape[-1] != 2:
            raise ValidationError("Keypoint arrays must have size-2 dimensions")

    @property  # Overriding
    def scale(self) -> FloatArray:
        """Scaling array."""
        if self._scale is None or len(self._scale) == 4:
            width = self.image_width
            height = self.image_height
            self._scale = np.array([width, height])
        return self._scale

    @property
    def is_relative(self) -> bool:
        """Parameterization is relative."""
        return bool(self.ptype.is_relative)

    @property
    def is_absolute(self) -> bool:
        """Parameterization is absolute."""
        return bool(self.ptype.is_absolute)

    def to_relative(self: T) -> T:
        """Convert kp to relative pixel coordinates, if necessary."""
        if self.is_relative:
            return self
        x = self.array.copy()
        x[..., :2] /= self.scale
        return self.__class__(
            x,
            ptype=self.ptype.as_relative,
            **self.metadata_kwargs,
        )

    def to_absolute(self: T) -> T:
        """Convert kp to absolute pixel coordinates, if necessary."""
        if self.is_absolute:
            return self
        x = self.array.copy()
        x[..., :2] *= self.scale
        return self.__class__(
            x,
            ptype=self.ptype.as_absolute,
            **self.metadata_kwargs,
        )  # Note to self: Make sure these two works right not sure yet

    def validate_known_ptype(self) -> None:
        """Make sure PType is a known box parameterization."""
        known_box_parameterizations = {
            PType.absolute_kp,
            PType.relative_kp,
            PType.heatmap,
        }
        if self.ptype not in known_box_parameterizations:
            raise ValidationError(f"Unknown box parameterization: {self.ptype.name}")

    @property
    def x(self) -> FloatArray:
        """Slice the x dimension of the keypoint."""
        self.validate_known_ptype()
        result: FloatArray
        result = self.array[..., 0]
        return result

    @property
    def y(self) -> FloatArray:
        """Slice the y dimension of the keypoint."""
        self.validate_known_ptype()
        result: FloatArray
        result = self.array[..., 1]
        return result

    def to_heatmap_array(self, covariance: float = 20) -> np.ndarray:
        """Create a 2D heatmap from an x, y pixel location."""
        xs = self.array[..., 0]
        ys = self.array[..., 1]
        img_w = self.image_width
        img_h = self.image_height
        heatmaps = []
        for x0, y0 in zip(xs, ys):
            xx, yy = np.meshgrid(np.arange(img_w), np.arange(img_h))
            zz = (
                1
                / (2 * np.pi * covariance**2)
                * np.exp(
                    -(
                        (xx - x0) ** 2 / (2 * covariance**2)
                        + (yy - y0) ** 2 / (2 * covariance**2)
                    )
                )
            )
            # Normalize zz to be in [0, 1]
            zz_min = zz.min()
            zz_max = zz.max()
            zz_range = zz_max - zz_min
            if zz_range == 0:
                zz_range += 1e-8
            heatmap = (zz - zz_min) / zz_range
            heatmaps.append(heatmap)
        return np.stack(heatmaps)
