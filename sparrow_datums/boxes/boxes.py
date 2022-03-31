from typing import TypeVar

import numpy as np

from ..chunk import Chunk
from ..types import FloatArray, PType

T = TypeVar("T", bound="Boxes")


class Boxes(Chunk):
    """
    Dense data arrays for boxes.

    This mostly serves as a base class for more specific types of box chunks.
    It inherits from :class:`.Chunk`.
    The underlying NumPy array should have shape ``(..., 4)``.

    Parameters
    ----------
    data : FloatArray
        A numpy array of dense floats
    ptype : PType
        The parameterization of the dense data
    image_width : int, optional
        The width of the relevant image
    image_height : int, optional
        The height of the relevant image

    Example
    -------
    >>> import numpy as np
    >>> from sparrow_datums import Boxes, PType
    >>> Boxes(np.ones(4), PType.absolute_tlbr).to_tlwh()
    Boxes([1., 1., 0., 0.])
    """

    def validate(self) -> None:
        """Check validity of boxes array."""
        if not self.shape or self.shape[-1] != 4:
            raise ValueError("Box arrays must have size-4 dimensions")

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

    def box_resize(self: T, image_width: int, image_height: int) -> T:
        """Resize boxes to a new width and height."""
        relative_boxes = self.to_relative()
        new_scale = np.array([image_width, image_height, image_width, image_height])
        x = relative_boxes.array.copy()
        x[..., :4] *= new_scale
        metadata_kwargs = self.metadata_kwargs.copy()
        metadata_kwargs["image_width"] = image_width
        metadata_kwargs["image_height"] = image_height
        return self.__class__(
            x,
            ptype=self.ptype.as_absolute,
            **metadata_kwargs,
        )

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

    def validate_known_ptype(self) -> None:
        """Make sure PType is a known box parameterization."""
        known_box_parameterizations = {
            PType.absolute_tlbr,
            PType.absolute_tlwh,
            PType.relative_tlbr,
            PType.relative_tlwh,
        }
        if self.ptype not in known_box_parameterizations:
            raise ValueError(f"Unknown box parameterization: {self.ptype.name}")

    @property
    def x(self) -> FloatArray:
        """Slice the x dimension of the boxes."""
        self.validate_known_ptype()
        result: FloatArray
        result = self.array[..., 0]
        return result

    @property
    def y(self) -> FloatArray:
        """Slice the y dimension of the boxes."""
        self.validate_known_ptype()
        result: FloatArray
        result = self.array[..., 1]
        return result

    @property
    def w(self) -> FloatArray:
        """Slice the width dimension of the boxes."""
        self.validate_known_ptype()
        result: FloatArray
        if self.is_tlwh:
            result = self.array[..., 2]
            return result
        result = self.array[..., 2] - self.array[..., 0]
        return result

    @property
    def h(self) -> FloatArray:
        """Slice the height dimension of the boxes."""
        self.validate_known_ptype()
        result: FloatArray
        if self.is_tlwh:
            result = self.array[..., 3]
            return result
        result = self.array[..., 3] - self.array[..., 1]
        return result

    @property
    def x1(self) -> FloatArray:
        """Slice the x1 dimension of the boxes."""
        self.validate_known_ptype()
        return self.x

    @property
    def y1(self) -> FloatArray:
        """Slice the y1 dimension of the boxes."""
        self.validate_known_ptype()
        return self.y

    @property
    def x2(self) -> FloatArray:
        """Slice the x2 dimension of the boxes."""
        self.validate_known_ptype()
        result: FloatArray
        if self.is_tlbr:
            result = self.array[..., 2]
            return result
        result = self.array[..., 0] + self.array[..., 2]
        return result

    @property
    def y2(self) -> FloatArray:
        """Slice the y2 dimension of the boxes."""
        self.validate_known_ptype()
        result: FloatArray
        if self.is_tlbr:
            result = self.array[..., 3]
            return result
        result = self.array[..., 1] + self.array[..., 3]
        return result
