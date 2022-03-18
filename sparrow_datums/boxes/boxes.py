from typing import Optional

import numpy as np

from ..chunk import Chunk
from .types import BoxType


class Boxes(Chunk):
    """
    Adds box_type to a NumPy array for strictness checking

    Parameters
    ----------
    data : np.ndarray
        A (..., 4) array of boxes
    type : BoxType, optional
        The parameterization of the boxes
    image_width : float, optional
        The width of the image
    image_height : float, optional
        The height of the image
    fps : float, optional
        The framerate if the boxes are being tracked
    object_ids : List[str], optional
        Tracking IDs for the objects
    """

    def check_shape(self) -> None:
        if not self.shape or self.shape[-1] != 4:
            raise ValueError("Box arrays must have size-4 dimensions")

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
        return self.type.is_relative

    @property
    def is_absolute(self) -> bool:
        return self.type.is_absolute

    @property
    def is_tlbr(self) -> bool:
        return self.type.is_tlbr

    @property
    def is_tlwh(self) -> bool:
        return self.type.is_tlwh

    def to_relative(self) -> "Boxes":
        """Convert boxes to relative pixel coordinates, if necessary"""
        if self.type.is_relative:
            return self
        return self.__class__(
            self / self.scale,
            type=self.type.as_relative,
            **self.metadata_kwargs,
        )

    def to_absolute(self) -> "Boxes":
        """Convert boxes to absolute pixel coordinates, if necessary"""
        if self.type.is_absolute:
            return self
        return self.__class__(
            self * self.scale,
            type=self.type.as_absolute,
            **self.metadata_kwargs,
        )

    def to_tlbr(self) -> "Boxes":
        """Convert boxes to tlbr format, if necessary"""
        if self.type.is_tlbr:
            return self
        x = self.array[..., 0]
        y = self.array[..., 1]
        w = self.array[..., 2]
        h = self.array[..., 3]
        return self.__class__(
            np.stack([x, y, x + w, y + h], -1),
            type=self.type.as_tlbr,
            **self.metadata_kwargs,
        )

    def to_tlwh(self) -> "Boxes":
        if self.type.is_tlwh:
            return self
        x1 = self.array[..., 0]
        y1 = self.array[..., 1]
        x2 = self.array[..., 2]
        y2 = self.array[..., 3]
        return self.__class__(
            np.stack([x1, y1, x2 - x1, y2 - y1], -1),
            type=self.type.as_tlwh,
            **self.metadata_kwargs,
        )


class SingleBox(Boxes):
    def check_shape(self) -> None:
        super().check_shape()
        if self.ndim > 1:
            raise ValueError("Single box must be a 1D array")


class FrameBoxes(Boxes):
    def check_shape(self) -> None:
        super().check_shape()
        if self.ndim != 2:
            raise ValueError("FrameBoxes should be (n_boxes, 4)")
