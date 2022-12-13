from typing import TypeVar

import numpy as np

from ..chunk import Chunk
from ..exceptions import ValidationError
from .keypoint_types import FloatArray, PType

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
    >>> Keypoints(np.ones(2), PType.absolute_tlbr).to_tlwh()
    Keypoints([1., 1.])
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        if not self.shape or self.shape[-1] != 2:
            raise ValidationError("Keypoint arrays must have size-2 dimensions")

    @property
    def is_relative(self) -> bool:
        """Parameterization is relative."""
        return bool(self.ptype.is_relative)

    @property
    def is_absolute(self) -> bool:
        """Parameterization is absolute."""
        return bool(self.ptype.is_absolute)

    # @TODO
    def to_relative(self: T) -> T:
        """Convert boxes to relative pixel coordinates, if necessary."""
        if self.is_relative:
            return self
        x = self.array.copy()
        x[..., :2] /= self.scale
        return self.__class__(
            x,
            ptype=self.ptype.as_relative,
            **self.metadata_kwargs,
        )  # Note to self: Make sure these two works right not sure yet

    # @TODO
    def to_absolute(self: T) -> T:
        """Convert boxes to absolute pixel coordinates, if necessary."""
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
        known_box_parameterizations = {PType.unknown}
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
