from typing import TypeVar

import numpy as np
import numpy.typing as npt

from ..chunk import Chunk

T = TypeVar("T", bound="Boxes")


class Boxes(Chunk):
    """Adds box_type to a NumPy array for strictness checking."""

    def validate(self) -> None:
        """Check validity of boxes array."""
        if not self.shape or self.shape[-1] != 4:
            raise ValueError("Box arrays must have size-4 dimensions")
        if np.any(self.array < 0):
            raise ValueError("Negative box values are not allowed")

    @property
    def is_relative(self) -> bool:
        """Parameterization is relative."""
        return bool(self.ptype.is_relative)

    @property
    def is_absolute(self) -> bool:
        """Parameterization is absolute."""
        return bool(self.ptype.is_absolute)

    @property
    def is_tlbr(self) -> bool:
        """Parameterization is TLBR."""
        return bool(self.ptype.is_tlbr)

    @property
    def is_tlwh(self) -> bool:
        """Parameterization is TLWH."""
        return bool(self.ptype.is_tlwh)

    def to_relative(self: T) -> T:
        """Convert boxes to relative pixel coordinates, if necessary."""
        if self.is_relative:
            return self
        x = self.array.copy()
        x[..., :4] /= self.scale
        return self.__class__(
            x,
            ptype=self.ptype.as_relative,
            **self.metadata_kwargs,
        )

    def to_absolute(self: T) -> T:
        """Convert boxes to absolute pixel coordinates, if necessary."""
        if self.is_absolute:
            return self
        x = self.array.copy()
        x[..., :4] *= self.scale
        return self.__class__(
            x,
            ptype=self.ptype.as_absolute,
            **self.metadata_kwargs,
        )

    def to_tlbr(self: T) -> T:
        """Convert boxes to TLBR format, if necessary."""
        if self.is_tlbr:
            return self
        x = self.array[..., 0]
        y = self.array[..., 1]
        w = self.array[..., 2]
        h = self.array[..., 3]
        return self.__class__(
            np.concatenate(
                [np.stack([x, y, x + w, y + h], -1), self.array[..., 4:]], -1
            ),
            ptype=self.ptype.as_tlbr,
            **self.metadata_kwargs,
        )

    def to_tlwh(self: T) -> T:
        """Convert boxes to TLWH format, if necessary."""
        if self.is_tlwh:
            return self
        x1 = self.array[..., 0]
        y1 = self.array[..., 1]
        x2 = self.array[..., 2]
        y2 = self.array[..., 3]
        return self.__class__(
            np.concatenate(
                [np.stack([x1, y1, x2 - x1, y2 - y1], -1), self.array[..., 4:]], -1
            ),
            ptype=self.ptype.as_tlwh,
            **self.metadata_kwargs,
        )

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """Slice the x dimension of the boxes."""
        result: npt.NDArray[np.float64]
        result = self.array[..., 0]
        return result

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """Slice the y dimension of the boxes."""
        result: npt.NDArray[np.float64]
        result = self.array[..., 1]
        return result

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """Slice the width dimension of the boxes."""
        result: npt.NDArray[np.float64]
        if self.is_tlwh:
            result = self.array[..., 2]
            return result
        if np.any(self.array[..., 0] > self.array[..., 2]):
            raise ValueError("x2 must be >= x1 for all boxes")
        result = self.array[..., 2] - self.array[..., 0]
        return result

    @property
    def h(self) -> npt.NDArray[np.float64]:
        """Slice the height dimension of the boxes."""
        result: npt.NDArray[np.float64]
        if self.is_tlwh:
            result = self.array[..., 3]
            return result
        if np.any(self.array[..., 1] > self.array[..., 3]):
            raise ValueError("y2 must >= y1 for all boxes")
        result = self.array[..., 3] - self.array[..., 1]
        return result

    @property
    def x1(self) -> npt.NDArray[np.float64]:
        """Slice the x1 dimension of the boxes."""
        return self.x

    @property
    def y1(self) -> npt.NDArray[np.float64]:
        """Slice the y1 dimension of the boxes."""
        return self.y

    @property
    def x2(self) -> npt.NDArray[np.float64]:
        """Slice the x2 dimension of the boxes."""
        result: npt.NDArray[np.float64]
        if self.is_tlbr:
            result = self.array[..., 2]
            return result
        result = self.array[..., 0] + self.array[..., 2]
        return result

    @property
    def y2(self) -> npt.NDArray[np.float64]:
        """Slice the y2 dimension of the boxes."""
        result: npt.NDArray[np.float64]
        if self.is_tlbr:
            result = self.array[..., 3]
            return result
        result = self.array[..., 1] + self.array[..., 3]
        return result
