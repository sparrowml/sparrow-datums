from typing import Optional

import numpy as np

from .box_types import BoxType


class Boxes(np.ndarray):
    """
    Adds box_type to a NumPy array for strictness checking

    Parameters
    ----------
    boxes : np.ndarray
        A (..., 4) array of boxes
    box_type : BoxType
        The parameterization of the boxes
    image_width : float, optional
        The width of the image
    image_height : float, optional
        The height of the image
    """

    def __new__(
        cls,
        boxes: np.ndarray,
        box_type: Optional[BoxType] = None,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> None:
        cls._box_type = box_type
        cls._image_width = image_width
        cls._image_height = image_height
        cls._scale = None
        return super().__new__(cls, boxes.shape, dtype=boxes.dtype, buffer=boxes.data)

    def __init__(self, *args, **kwargs) -> None:
        """ndarray subclasses don't need __init__, but pylance does"""
        pass

    def __array_finalize__(self, _) -> None:
        if not getattr(self, "shape", None) or self.shape[-1] != 4:
            raise ValueError("Box arrays must have 4 dimensions")

    @property
    def array(self) -> np.ndarray:
        return self.view(np.ndarray)

    @property
    def image_width(self) -> float:
        if self._image_width is None:
            raise ValueError("image_width not set")
        return self._image_width

    @property
    def image_height(self) -> float:
        if self._image_height is None:
            raise ValueError("image_height not set")
        return self._image_height

    @property
    def scale(self) -> np.ndarray:
        """Convert boxes to relative pixel coordinates, if necessary"""
        if self._scale is None:
            width = self.image_width
            height = self.image_height
            self._scale = np.array([width, height, width, height])
        return self._scale

    @property
    def box_type(self) -> BoxType:
        if self._box_type is None:
            raise TypeError("Boxes instance is not typed")
        return self._box_type

    @property
    def is_relative(self) -> bool:
        return self.box_type.is_relative

    def to_relative(self) -> "Boxes":
        """Convert boxes to relative pixel coordinates, if necessary"""
        if self.box_type.is_relative:
            return self
        return Boxes(
            self / self.scale,
            box_type=self.box_type.as_relative,
            image_width=self._image_width,
            image_height=self._image_height,
        )

    @property
    def is_absolute(self) -> bool:
        return self.box_type.is_absolute

    def to_absolute(self) -> "Boxes":
        """Convert boxes to absolute pixel coordinates, if necessary"""
        if self.box_type.is_absolute:
            return self
        return Boxes(
            self * self.scale,
            box_type=self.box_type.as_absolute,
            image_width=self._image_width,
            image_height=self._image_height,
        )
