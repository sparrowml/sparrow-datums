from typing import Optional

import numpy as np

from sparrow_datums.types import BoxType
from sparrow_datums.chunk import Chunk


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

    def to_relative(self) -> "Boxes":
        """Convert boxes to relative pixel coordinates, if necessary"""
        if self.type.is_relative:
            return self
        return Boxes(
            self / self.scale,
            type=self.type.as_relative,
            image_width=self._image_width,
            image_height=self._image_height,
        )

    @property
    def is_absolute(self) -> bool:
        return self.type.is_absolute

    def to_absolute(self) -> "Boxes":
        """Convert boxes to absolute pixel coordinates, if necessary"""
        if self.type.is_absolute:
            return self
        return Boxes(
            self * self.scale,
            type=self.type.as_absolute,
            image_width=self._image_width,
            image_height=self._image_height,
        )
