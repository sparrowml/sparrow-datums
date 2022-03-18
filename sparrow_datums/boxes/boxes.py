from typing import Optional

from multiprocessing.sharedctypes import Value

import numpy as np
import numpy.typing as npt

from ..chunk import Chunk
from .types import BoxType


class Boxes(Chunk):
    """Adds box_type to a NumPy array for strictness checking"""

    def validate(self) -> None:
        if not self.shape or self.shape[-1] != 4:
            raise ValueError("Box arrays must have size-4 dimensions")
        if np.any(self.array < 0):
            raise ValueError("Negative box values are not allowed")

    @property
    def type(self) -> BoxType:
        _type: BoxType = self._type
        return _type

    @classmethod
    def decode_type(cls, type_name: Optional[str]) -> Optional[BoxType]:
        if type_name is None:
            return None
        return BoxType(type_name)

    @property
    def is_relative(self) -> bool:
        return bool(self.type.is_relative)

    @property
    def is_absolute(self) -> bool:
        return bool(self.type.is_absolute)

    @property
    def is_tlbr(self) -> bool:
        return bool(self.type.is_tlbr)

    @property
    def is_tlwh(self) -> bool:
        return bool(self.type.is_tlwh)

    def to_relative(self) -> "Boxes":
        """Convert boxes to relative pixel coordinates, if necessary"""
        if self.is_relative:
            return self
        x = self.array.copy()
        x[..., :4] /= self.scale
        return self.__class__(
            x,
            type=self.type.as_relative,
            **self.metadata_kwargs,
        )

    def to_absolute(self) -> "Boxes":
        """Convert boxes to absolute pixel coordinates, if necessary"""
        if self.is_absolute:
            return self
        x = self.array.copy()
        x[..., :4] *= self.scale
        return self.__class__(
            x,
            type=self.type.as_absolute,
            **self.metadata_kwargs,
        )

    def to_tlbr(self) -> "Boxes":
        """Convert boxes to tlbr format, if necessary"""
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
            type=self.type.as_tlbr,
            **self.metadata_kwargs,
        )

    def to_tlwh(self) -> "Boxes":
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
            type=self.type.as_tlwh,
            **self.metadata_kwargs,
        )

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.array[..., 0]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self.array[..., 1]

    @property
    def w(self) -> npt.NDArray[np.float64]:
        if self.is_tlwh:
            return self.array[..., 2]
        if np.any(self.array[..., 0] > self.array[..., 2]):
            raise ValueError("x2 must be >= x1 for all boxes")
        return self.array[..., 2] - self.array[..., 0]

    @property
    def h(self) -> npt.NDArray[np.float64]:
        if self.is_tlwh:
            return self.array[..., 3]
        if np.any(self.array[..., 1] > self.array[..., 3]):
            raise ValueError("y2 must >= y1 for all boxes")
        return self.array[..., 3] - self.array[..., 1]

    @property
    def x1(self) -> npt.NDArray[np.float64]:
        return self.x

    @property
    def y1(self) -> npt.NDArray[np.float64]:
        return self.y

    @property
    def x2(self) -> npt.NDArray[np.float64]:
        if self.is_tlbr:
            return self.array[..., 2]
        return self.array[..., 0] + self.array[..., 2]

    @property
    def y2(self) -> npt.NDArray[np.float64]:
        if self.is_tlbr:
            return self.array[..., 3]
        return self.array[..., 1] + self.array[..., 3]
