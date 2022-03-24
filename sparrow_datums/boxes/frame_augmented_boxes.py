import json
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np
import numpy.typing as npt

from ..types import PType
from .augmented_boxes import AugmentedBoxes
from .frame_boxes import _is_2d
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
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
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
        label_names: Optional[List[str]] = None,
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
        cls: Type["FrameAugmentedBoxes"],
        darwin_dict: Dict[str, Any],
        label_names: List[str] = [],
    ) -> "FrameAugmentedBoxes":
        """Create FrameAugmentedBoxes from a serialized Darwin dict."""
        label_names_map = {name: float(idx) for idx, name in enumerate(label_names)}
        image_width, image_height = itemgetter("width", "height")(darwin_dict["image"])
        boxes = []
        score = 1.0
        for annotation in darwin_dict["annotations"]:
            x, y, w, h = itemgetter("x", "y", "w", "h")(annotation["bounding_box"])
            label = label_names_map.get(annotation["name"], -1.0)
            boxes.append([x, y, w, h, score, label])
        data: npt.NDArray[np.float64] = np.array(boxes).astype("float64")
        return cls(
            data,
            ptype=PType.absolute_tlwh,
            image_width=image_width,
            image_height=image_height,
        )

    @classmethod
    def from_darwin_file(
        cls: Type["FrameAugmentedBoxes"],
        path: Union[str, Path],
        label_names: List[str] = [],
    ) -> "FrameAugmentedBoxes":
        """Read FrameAugmentedBoxes from Darwin dict on disk."""
        with open(path) as f:
            darwin_dict = json.loads(f.read())
        return cls.from_darwin_dict(darwin_dict, label_names=label_names)