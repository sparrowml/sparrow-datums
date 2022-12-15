from typing import TypeVar

import numpy as np

from ..chunk import Chunk
from ..exceptions import ValidationError
from ..types import FloatArray, PType

T = TypeVar("T", bound="Keypoints")


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
        """Check validity of keypoints array."""
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
        """Convert keypoint (kp) to relative pixel coordinates, if necessary."""
        if self.is_relative:
            return self
        x = self.array.copy().astype(np.float32)
        x[..., :2] /= self.scale
        if self.ptype is PType.unknown:
            param_type = PType.relative_kp
        else:
            param_type = self.ptype.as_relative
        return self.__class__(
            x,
            ptype=param_type,
            **self.metadata_kwargs,
        )

    def to_absolute(self: T) -> T:
        """Convert keypoint (kp) to absolute pixel coordinates, if necessary."""
        if self.is_absolute:
            return self
        x = self.array.copy().astype(np.float32)
        x[..., :2] *= self.scale
        if self.ptype is PType.unknown:
            param_type = PType.absolute_kp
        else:
            param_type = self.ptype.as_absolute
        return self.__class__(
            x,
            ptype=param_type,
            **self.metadata_kwargs,
        )

    def validate_known_ptype(self) -> None:
        """Make sure PType is a known keypoint parameterization."""
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

    def generate_heatmap(self, x0, y0, covariance):
        """Create a 2D heatmap from an x, y pixel location.

        Note: This is a helper function for to_heatmap_array()

        Parameters
        ----------
        x0 : int
            x coordinate of the keypoint to be transformed
        y0 : int
            y coordinate of the keypoint to be transformed
        covariance : float
            covariance of the surface to be created

        Returns
        -------
        np.ndarray
            keypoint in form of a 2D array( a heatmap)
        """
        img_w = self.image_width
        img_h = self.image_height
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
        return (zz - zz_min) / zz_range

    def to_heatmap_array(self, covariance=20):
        """Convert Keypoints into a heatmap array.

        Parameters
        ----------
        covariance : float, optional
            Covariance of the surface to be created, by default 20

        Returns
        -------
        np.ndarray
            heatmaps in numpy form.
        """
        xs = self.array[..., 0]
        ys = self.array[..., 1]
        if np.size(xs) > 1:
            heatmaps = []
            for x0, y0 in zip(xs, ys):
                heatmap = self.generate_heatmap(x0, y0, covariance)
                heatmaps.append(heatmap)
            return np.stack(heatmaps)
        elif np.size(xs) == 1:
            return self.generate_heatmap(xs.item(), ys.item(), covariance)
