"""BoxTracking chunk."""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from ..boxes import Boxes, FrameBoxes
from ..types import PType
from .tracking import Tracking


class BoxTracking(Tracking, Boxes):
    """
    Dense data arrays for box tracking.

    It inherits from :class:`.Boxes`.
    The underlying NumPy array should have shape ``(n_frames, n_objects, 4)``.

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
    fps : float, optional
        The framerate of the chunk data (if tracking)
    object_ids:  list[str], optional
        Identifiers for the objects (if tracking)
    """

    def validate(self) -> None:
        """Validate tracking shape and boxes."""
        super().validate()
        Boxes.validate(self)

    def __iter__(self) -> Iterator[FrameBoxes]:
        """Yield FrameBoxes objects for each frame."""
        for box in self.view(Boxes):
            yield box.view(FrameBoxes)

    @classmethod
    def from_frame_boxes(
        cls: type["BoxTracking"],
        boxes: list[FrameBoxes],
        ptype: PType = PType.unknown,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        fps: Optional[float] = None,
        object_ids: Optional[list[str]] = None,
    ) -> "BoxTracking":
        """
        Create an BoxTracking chunk from a list of FrameBoxes objects.

        This is typically used for storing detections on a video.
        """
        max_boxes = max(map(len, boxes))
        data = np.zeros((len(boxes), max_boxes, 4)) * np.nan
        for i, frame in enumerate(boxes):
            data[i, : len(frame)] = frame.array
        return cls(
            data,
            ptype=ptype,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            object_ids=object_ids,
        )
