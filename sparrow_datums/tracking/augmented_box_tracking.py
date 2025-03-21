"""AugmentedBoxTracking chunk."""

from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt

from ..boxes import AugmentedBoxes, FrameAugmentedBoxes
from ..chunk_types import PType
from .box_tracking import BoxTracking
from .tracking import Tracking


class AugmentedBoxTracking(Tracking, AugmentedBoxes):
    """
    Dense data arrays for box tracking with [boxes, scores, labels].

    It inherits from :class:`.AugmentedBoxes`.
    The underlying NumPy array should have shape ``(n_frames, n_objects, 6)``.

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
    object_ids:  List[str], optional
        Identifiers for the objects (if tracking)
    """

    empty_shape: tuple[int, int, int] = (0, 0, 6)

    def validate(self) -> None:
        """Validate tracking shape and boxes."""
        super().validate()
        AugmentedBoxes.validate(self)

    def __iter__(self) -> Iterator[FrameAugmentedBoxes]:
        """Yield FrameAugmentedBoxes objects for each frame."""
        for box in self.view(AugmentedBoxes):
            yield box.view(FrameAugmentedBoxes)

    @classmethod
    def from_cvat(
        cls: type["AugmentedBoxTracking"], annotations_path: Union[str, Path]
    ) -> "AugmentedBoxTracking":
        """Create AugmentedBoxTracking from a CVAT video file."""
        raise NotImplementedError

    @classmethod
    def from_dict(
        cls: type["AugmentedBoxTracking"],
        chunk_dict: dict[str, Any],
        dims: Optional[int] = None,
    ) -> "AugmentedBoxTracking":
        """Create chunk from chunk dict."""
        return super().from_dict(chunk_dict, dims=6)

    @classmethod
    def from_frame_augmented_boxes(
        cls: type["AugmentedBoxTracking"],
        boxes: list[FrameAugmentedBoxes],
        ptype: PType = PType.unknown,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        fps: Optional[float] = None,
        object_ids: Optional[list[str]] = None,
    ) -> "AugmentedBoxTracking":
        """
        Create an AugmentedBoxTracking chunk from a list of FrameAugmentedBoxes objects.

        This is typically used for storing detections on a video.
        """
        max_boxes = max(map(len, boxes))
        data = np.zeros((len(boxes), max_boxes, 6)) * np.nan
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

    def filter_by_class(self, class_idx: int) -> BoxTracking:
        """Get a single class and return a new BoxTracking chunk."""
        object_mask: list[int] = []
        for j in range(self.shape[1]):
            if class_idx in set(self.array[:, j, -1].ravel()):
                object_mask.append(j)
        if len(object_mask) > 0:
            data = self.array[:, np.array(object_mask), :4]
        else:
            data = np.zeros((len(self), 0, 4)) * np.nan
        return BoxTracking(
            data,
            ptype=self.ptype,
            **self.metadata_kwargs,
        )

    @classmethod
    def from_box_tracking(
        cls: type["AugmentedBoxTracking"],
        chunk: BoxTracking,
        score: float = 0.0,
        class_idx: int = 0,
    ) -> "AugmentedBoxTracking":
        """Create AugmentedBoxTracking chunk from BoxTracking chunk."""
        pad_shape = np.ones(chunk.shape[:2] + (1,))
        data = np.concatenate(
            [
                chunk.array[..., :4],
                pad_shape * score,
                pad_shape * class_idx,
            ],
            axis=-1,
        )
        return cls(data, ptype=chunk.ptype, **chunk.metadata_kwargs)
