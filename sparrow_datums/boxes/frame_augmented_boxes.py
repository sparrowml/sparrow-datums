from __future__ import annotations

import json
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt

from ..types import PType
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

    def validate(self) -> None:
        """Check validity of boxes array."""
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleAugmentedBox]:
        """Yield SingleAugmentedBox objects for each box."""
        for box in self.view(AugmentedBoxes):
            yield box.view(SingleAugmentedBox)

    def to_darwin_dict(
        self,
        filename: str,
        path: str = "/",
        label_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Serialize boxes to a Darwin annotation dict."""
        if label_names is None:
            label_names = ["Unknown"] * (self.labels.max() + 1)
        return {
            "image": {"filename": filename, "path": path},
            "annotations": [
                {
                    "bounding_box": {
                        "x": float(box.x),
                        "y": float(box.y),
                        "w": float(box.w),
                        "h": float(box.h),
                    },
                    "name": box.names(label_names),
                }
                for box in self.to_absolute()
            ],
        }

    def to_darwin_file(
        self,
        output_path: Union[str, Path],
        filename: str,
        path: str = "/",
        label_names: Optional[list[str]] = None,
    ) -> None:
        """
        Write Darwin annotation dict to disk.

        Parameters
        ----------
        output_path : str
            The full path (including filename) where the Darwin annotation will be written on disk
        filename : str
            The fimename of the image or video as it's referenced within the Darwin
        path : str, optional
            The path of the file within Darwin. The path starts at root ("/"),
            and expands including the folders in which the resource is stored within Darwin.
        label_names : list of str, optional
            The names of the labeled classes annotated in the image or video. If None,
            "Unknown" will be used.
        """
        with open(output_path, "w") as f:
            f.write(
                json.dumps(
                    self.to_darwin_dict(filename, path=path, label_names=label_names)
                )
            )

    @classmethod
    def from_darwin_dict(
        cls: type["FrameAugmentedBoxes"],
        darwin_dict: dict[str, Any],
        label_names: list[str] = [],
    ) -> "FrameAugmentedBoxes":
        """Create FrameAugmentedBoxes from a serialized Darwin dict."""
        label_names_map = {name: float(idx) for idx, name in enumerate(label_names)}
        image_width, image_height = itemgetter("width", "height")(darwin_dict["image"])
        boxes = []
        score = 1.0
        for annotation in darwin_dict["annotations"]:
            if "bounding_box" not in annotation:
                continue
            x, y, w, h = itemgetter("x", "y", "w", "h")(annotation["bounding_box"])
            label = label_names_map.get(annotation["name"], -1.0)
            boxes.append([x, y, w, h, score, label])
        data: npt.NDArray[np.float64]
        if len(boxes):
            data = np.array(boxes).astype("float64")
        else:
            data = np.zeros((0, 6), "float64")
        return cls(
            data,
            ptype=PType.absolute_tlwh,
            image_width=image_width,
            image_height=image_height,
        )

    @classmethod
    def from_darwin_file(
        cls: type["FrameAugmentedBoxes"],
        path: Union[str, Path],
        label_names: list[str] = [],
    ) -> "FrameAugmentedBoxes":
        """Read FrameAugmentedBoxes from Darwin dict on disk."""
        with open(path) as f:
            darwin_dict = json.loads(f.read())
        return cls.from_darwin_dict(darwin_dict, label_names=label_names)

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
            raise ValueError(
                "SingleAugmentedBox with different PType cannot be concatenated"
            )
        if self.metadata_kwargs != box.metadata_kwargs:
            raise ValueError(
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
