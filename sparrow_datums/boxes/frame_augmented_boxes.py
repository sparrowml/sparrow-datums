from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt

from ..chunk_types import PType
from ..exceptions import ValidationError
from .augmented_boxes import AugmentedBoxes
from .frame_boxes import FrameBoxes, _is_2d
from .single_augmented_box import SingleAugmentedBox


class FrameAugmentedBoxes(AugmentedBoxes):
    """
    2D dense data arrays for augmented boxes.

    The data contain ``[boxes, scores, labels]`` components.
    It inherits from :class:`.AugmentedBoxes`.
    The underlying NumPy array should have shape ``(n_boxes, 6)``,
    with 4 dimensions reserved for boxes, and the last two for
    scores and labels respectively.

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
    >>> from sparrow_datums import FrameAugmentedBoxes, PType
    >>> augmented_boxes = FrameAugmentedBoxes(np.array([[0, 0, 1, 1, 0.5, 3]]))
    >>> for box in augmented_boxes: print(box)
    [0.  0.  1.  1.  0.5 3. ]
    """

    empty_shape: tuple[int, int] = (0, 6)

    def validate(self) -> None:
        """Check validity of boxes array."""
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleAugmentedBox]:
        """Yield SingleAugmentedBox objects for each box."""
        for box in self.view(AugmentedBoxes):
            yield box.view(SingleAugmentedBox)

    @classmethod
    def from_dict(
        cls: type["FrameAugmentedBoxes"],
        chunk_dict: dict[str, Any],
        dims: Optional[int] = None,
    ) -> "FrameAugmentedBoxes":
        """Create chunk from chunk dict."""
        return super().from_dict(chunk_dict, dims=6)

    @classmethod
    def from_single_box(
        cls: type["FrameAugmentedBoxes"], box: SingleAugmentedBox
    ) -> "FrameAugmentedBoxes":
        """Create a FrameBoxes object from a SingleBox."""
        return cls(
            box.array[None, :],
            ptype=box.ptype,
            **box.metadata_kwargs,
        )

    def add_box(self, box: SingleAugmentedBox) -> "FrameAugmentedBoxes":
        """Concatenate a single augmented box."""
        if self.ptype != box.ptype:
            raise ValidationError(
                "SingleAugmentedBox with different PType cannot be concatenated"
            )
        if self.metadata_kwargs != box.metadata_kwargs:
            raise ValidationError(
                "SingleAugmentedBox with different metadata cannot be concatenated"
            )
        return FrameAugmentedBoxes(
            np.concatenate([self.array, box.array[None]]),
            ptype=self.ptype,
            **self.metadata_kwargs,
        )

    def get_single_box(self, i: int) -> SingleAugmentedBox:
        """Get the ith element as a SingleAugmentedBox."""
        result: SingleAugmentedBox = self.view(AugmentedBoxes)[i].view(
            SingleAugmentedBox
        )
        return result

    def to_frame_boxes(self) -> FrameBoxes:
        """Drop augmented part of the data."""
        return FrameBoxes(
            self.array[..., :4],
            ptype=self.ptype,
            **self.metadata_kwargs,
        )
