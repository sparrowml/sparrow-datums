from typing import Any, Dict, Iterator, List, Optional, Union

import json
from pathlib import Path

import numpy as np

from .augmented_boxes import AugmentedBoxes
from .boxes import Boxes
from .single_box import SingleAugmentedBox, SingleBox


def _is_2d(x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError("A frame boxes object must be a 2D array")


class FrameBoxes(Boxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleBox]:
        for box in self.view(Boxes):
            yield box.view(SingleBox)


class FrameAugmentedBoxes(AugmentedBoxes):
    def validate(self) -> None:
        super().validate()
        _is_2d(self)

    def __iter__(self) -> Iterator[SingleAugmentedBox]:
        for box in self.view(AugmentedBoxes):
            yield box.view(SingleAugmentedBox)

    def to_darwin_dict(
        self,
        filename: str,
        path: str = "/",
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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

    def to_darwin_annotation_file(
        self,
        output_path: Union[str, Path],
        filename: str,
        path: str = "/",
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        with open(output_path, "w") as f:
            f.write(
                json.dumps(
                    self.to_darwin_dict(filename, path=path, label_names=label_names)
                )
            )
