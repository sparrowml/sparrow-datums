import numpy as np
import pytest

from sparrow_datums.boxes.frame_augmented_boxes import FrameAugmentedBoxes
from sparrow_datums.tracking.box_tracking import BoxTracking

from ..chunk_types import PType
from ..exceptions import ValidationError
from .augmented_box_tracking import AugmentedBoxTracking


def test_box_tracking_chunk_requires_3_dimensions():
    with pytest.raises(ValidationError):
        AugmentedBoxTracking(np.ones((2, 6)))
    with pytest.raises(ValidationError):
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


def test_from_frame_augmented_boxes():
    boxes_a = FrameAugmentedBoxes(np.ones((2, 6)), PType.absolute_tlwh)
    boxes_b = FrameAugmentedBoxes(np.ones((3, 6)), PType.absolute_tlwh)
    chunk = AugmentedBoxTracking.from_frame_augmented_boxes(
        [boxes_a, boxes_b], ptype=PType.absolute_tlwh
    )
    assert chunk.shape == (2, 3, 6)


def test_filter_by_class():
    # Create tracking data with different classes
    data = np.zeros((3, 3, 6))
    # Object 0 is class 0
    data[:, 0, -1] = 0
    # Object 1 is class 1
    data[:, 1, -1] = 1
    # Object 2 is class 0
    data[:, 2, -1] = 0

    chunk = AugmentedBoxTracking(data, ptype=PType.absolute_tlwh)
    cars = chunk.filter_by_class(0)
    bikes = chunk.filter_by_class(1)
    empty = chunk.filter_by_class(2)

    assert isinstance(cars, BoxTracking)
    assert isinstance(bikes, BoxTracking)
    assert isinstance(empty, BoxTracking)
    assert cars.shape[1] == 2  # 2 objects of class 0
    assert bikes.shape[1] == 1  # 1 object of class 1
    assert empty.shape[1] == 0  # 0 objects of class 2


def test_from_box_tracking():
    # Create a BoxTracking object
    box_data = np.ones((3, 2, 4))
    box_tracking = BoxTracking(box_data, ptype=PType.absolute_tlwh)

    # Convert to AugmentedBoxTracking
    augmented_tracking = AugmentedBoxTracking.from_box_tracking(
        box_tracking, score=1.0, class_idx=0
    )

    assert isinstance(augmented_tracking, AugmentedBoxTracking)
    np.testing.assert_equal(box_tracking.array, augmented_tracking.array[..., :4])
    assert augmented_tracking.shape == (3, 2, 6)
    assert (augmented_tracking.array[..., 4] == 1.0).all()  # All scores are 1.0
    assert (augmented_tracking.array[..., 5] == 0).all()  # All class indices are 0


def test_empty_chunk():
    data_dict = {
        "data": [[], [], []],
        "classname": "AugmentedBoxTracking",
        "ptype": "relative_tlbr",
        "image_width": 128,
        "image_height": 128,
        "fps": None,
        "object_ids": None,
        "start_time": None,
    }
    boxes = AugmentedBoxTracking.from_dict(data_dict)
    assert boxes.shape == (3, 0, 6)
