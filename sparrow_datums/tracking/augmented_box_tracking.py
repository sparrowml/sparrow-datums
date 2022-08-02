"""AugmentedBoxTracking chunk."""
from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt

from ..boxes import AugmentedBoxes, FrameAugmentedBoxes
from ..types import PType
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

    def validate(self) -> None:
        """Validate tracking shape and boxes."""
        super().validate()
        AugmentedBoxes.validate(self)

    def __iter__(self) -> Iterator[FrameAugmentedBoxes]:
        """Yield FrameAugmentedBoxes objects for each frame."""
        for box in self.view(AugmentedBoxes):
            yield box.view(FrameAugmentedBoxes)

    def to_darwin_dict(
        self,
        filename: str,
        path: str = "/",
        label_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Serialize boxes to a Darwin annotation dict."""
        if label_names is None:
            label_names = ["Unknown"] * (self.labels.max() + 1)

        annotations: list[dict[str, Any]] = [
            {"frames": {}} for _ in range(self.shape[1])
        ]
        for i, frame in enumerate(self.to_absolute()):
            for j, box in enumerate(frame):
                if np.isnan(box.array).any():
                    continue
                name = box.names(label_names)
                frame_dict = {
                    "bounding_box": {
                        "x": float(box.x),
                        "y": float(box.y),
                        "w": float(box.w),
                        "h": float(box.h),
                    },
                    "keyframe": True,
                }
                annotations[j]["name"] = name
                annotations[j]["frames"][str(i)] = frame_dict
                annotations[j]["id"] = self.object_ids[j]

        annotations = [x for x in annotations if len(x["frames"].keys()) != 0]

        for j in range(len(annotations)):
            frame_ids = list(map(int, annotations[j]["frames"].keys()))
            min_frame = min(frame_ids)
            max_frame = max(frame_ids) + 1
            annotations[j]["segments"] = [[min_frame, max_frame]]
        return {
            "image": {
                "filename": filename,
                "path": path,
                "fps": self.fps,
                "width": self.image_width,
                "height": self.image_height,
            },
            "annotations": annotations,
        }

    def to_darwin_file(
        self,
        output_path: Union[str, Path],
        filename: str,
        path: str = "/",
        label_names: Optional[list[str]] = None,
    ) -> None:
        """Write Darwin annotation dict to disk."""
        with open(output_path, "w") as f:
            f.write(
                json.dumps(
                    self.to_darwin_dict(filename, path=path, label_names=label_names)
                )
            )

    @classmethod
    def from_darwin_dict(
        cls: type["AugmentedBoxTracking"],
        darwin_dict: dict[str, Any],
        label_names: list[str] = [],
    ) -> "AugmentedBoxTracking":
        """Create AugmentedBoxTracking chunk from a serialized Darwin dict."""
        label_names_map = {name: float(idx) for idx, name in enumerate(label_names)}
        image_width = darwin_dict["image"]["width"]
        image_height = darwin_dict["image"]["height"]
        fps = darwin_dict["image"]["fps"]
        n_frames = darwin_dict["image"]["frame_count"]
        n_objects = len(darwin_dict["annotations"])
        data: npt.NDArray[np.float64] = np.zeros((n_frames, n_objects, 6)) * np.nan
        score = 1.0
        object_ids: list[str] = [t["id"] for t in darwin_dict["annotations"]]
        for i in range(n_frames):
            for j, tracklet in enumerate(darwin_dict["annotations"]):
                annotation = tracklet["frames"].get(str(i), {})
                if "bounding_box" not in annotation:
                    continue
                x, y, w, h = itemgetter("x", "y", "w", "h")(annotation["bounding_box"])
                label = label_names_map.get(tracklet["name"], -1.0)
                data[i, j] = x, y, w, h, score, label
        return cls(
            data,
            ptype=PType.absolute_tlwh,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            object_ids=object_ids,
        )

    @classmethod
    def from_darwin_file(
        cls: type["AugmentedBoxTracking"],
        path: Union[str, Path],
        label_names: list[str] = [],
    ) -> "AugmentedBoxTracking":
        """Read AugmentedBoxTracking from Darwin dict on disk."""
        with open(path) as f:
            darwin_dict = json.loads(f.read())
        return cls.from_darwin_dict(darwin_dict, label_names=label_names)

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
        return BoxTracking(
            self.array[:, np.array(object_mask), :4],
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
