import numpy as np
import pytest

from sparrow_datums.boxes.frame_augmented_boxes import FrameAugmentedBoxes
from sparrow_datums.tracking.box_tracking import BoxTracking

from ..types import PType
from .augmented_box_tracking import AugmentedBoxTracking

DARWIN_DICT = {
    "dataset": "Ring Tracking",
    "image": {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "original_filename": "Ring_541700402_1820_7075017446887328011.mp4",
        "filename": "Ring_541700402_1820_7075017446887328011.mp4",
        "url": "https://darwin.v7labs.com/api/videos/4007448/original",
        "path": "/",
        "workview_url": "https://darwin.v7labs.com/workview?dataset=470324&image=1",
        "frame_count": 3,
    },
    "annotations": [
        {
            "frames": {
                "0": {
                    "bounding_box": {"h": 32.1, "w": 60.35, "x": 704.11, "y": 414.68},
                    "keyframe": True,
                },
                "1": {
                    "bounding_box": {"h": 32.1, "w": 60.35, "x": 704.11, "y": 414.68},
                    "keyframe": False,
                },
                "2": {
                    "bounding_box": {"h": 32.1, "w": 60.35, "x": 704.11, "y": 414.68},
                    "keyframe": False,
                },
            },
            "id": "f90b983d-8d31-41c3-828f-69b993099ca7",
            "interpolate_algorithm": "linear-1.1",
            "interpolated": True,
            "name": "car",
            "segments": [[0, 3783]],
        },
        {
            "frames": {
                "0": {
                    "bounding_box": {"h": 16.05, "w": 35.31, "x": 659.81, "y": 398.62},
                    "keyframe": True,
                },
                "1": {
                    "bounding_box": {"h": 16.05, "w": 35.31, "x": 659.81, "y": 398.62},
                    "keyframe": False,
                },
                "2": {
                    "bounding_box": {"h": 16.05, "w": 35.31, "x": 659.81, "y": 398.62},
                    "keyframe": False,
                },
            },
            "id": "c2c5e6c7-f4f5-4edb-b24a-00aa76b7674e",
            "interpolate_algorithm": "linear-1.1",
            "interpolated": True,
            "name": "car",
            "segments": [[0, 3783]],
        },
        {
            "frames": {
                "0": {
                    "bounding_box": {
                        "h": 494.39,
                        "w": 559.11,
                        "x": 987.14,
                        "y": 494.43,
                    },
                    "keyframe": True,
                },
                "1": {
                    "bounding_box": {
                        "h": 494.39,
                        "w": 559.11,
                        "x": 987.14,
                        "y": 494.43,
                    },
                    "keyframe": False,
                },
                "2": {
                    "bounding_box": {
                        "h": 494.39,
                        "w": 559.11,
                        "x": 987.14,
                        "y": 494.43,
                    },
                    "keyframe": False,
                },
            },
            "id": "8ffd95e2-0641-445a-83e5-47c664edaaa5",
            "interpolate_algorithm": "linear-1.1",
            "interpolated": True,
            "name": "bicycle",
            "segments": [[0, 3783]],
        },
    ],
}


def test_box_tracking_chunk_requires_3_dimensions():
    with pytest.raises(ValueError):
        AugmentedBoxTracking(np.ones((2, 6)))
    with pytest.raises(ValueError):
        AugmentedBoxTracking(np.ones((2, 2, 4)))


def test_from_dict_with_empty_data():
    boxes = AugmentedBoxTracking.from_dict(
        {
            "data": [],
            "ptype": "unknown",
            "image_width": None,
            "image_height": None,
            "fps": None,
            "object_ids": None,
        }
    )
    assert isinstance(boxes, AugmentedBoxTracking)
    assert len(boxes) == 0


def test_frame_augmented_boxes_iterator_makes_frame_augmented_boxes():
    chunk = AugmentedBoxTracking(np.ones((2, 2, 6)), ptype=PType.absolute_tlwh)
    for frame in chunk:
        assert isinstance(frame, FrameAugmentedBoxes)
        assert frame.ptype == PType.absolute_tlwh


def test_from_darwin_dict_creates_augmented_box_tracking():
    chunk = AugmentedBoxTracking.from_darwin_dict(DARWIN_DICT)
    assert chunk.shape == (3, 3, 6)
    assert isinstance(chunk, AugmentedBoxTracking)
    assert chunk.ptype == PType.absolute_tlwh
    assert chunk.image_width
    assert chunk.image_height
    np.testing.assert_equal(chunk.labels, -1)
    assert "8ffd95e2-0641-445a-83e5-47c664edaaa5" in chunk.object_ids


def test_from_frame_augmented_boxes():
    boxes_a = FrameAugmentedBoxes(np.ones((2, 6)), PType.absolute_tlwh)
    boxes_b = FrameAugmentedBoxes(np.ones((3, 6)), PType.absolute_tlwh)
    chunk = AugmentedBoxTracking.from_frame_augmented_boxes(
        [boxes_a, boxes_b], ptype=PType.absolute_tlwh
    )
    assert chunk.shape == (2, 3, 6)


def test_filter_by_class():
    chunk = AugmentedBoxTracking.from_darwin_dict(
        DARWIN_DICT, label_names=["car", "bicycle"]
    )
    cars = chunk.filter_by_class(0)
    bikes = chunk.filter_by_class(1)
    assert isinstance(cars, BoxTracking)
    assert isinstance(bikes, BoxTracking)
    assert cars.shape[1] + bikes.shape[1] == chunk.shape[1]


def test_from_box_tracking():
    chunk = AugmentedBoxTracking.from_darwin_dict(
        DARWIN_DICT, label_names=["car", "bicycle"]
    )
    cars = chunk.filter_by_class(0)
    augmented_cars = AugmentedBoxTracking.from_box_tracking(
        cars, score=1.0, class_idx=0
    )
    assert isinstance(augmented_cars, AugmentedBoxTracking)
    np.testing.assert_equal(cars.array, augmented_cars.array[..., :4])
