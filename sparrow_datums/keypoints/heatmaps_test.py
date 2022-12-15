import doctest

import numpy as np
import pytest

from sparrow_datums.keypoints import heatmaps
from sparrow_datums.keypoints.heatmaps import Heatmaps
from sparrow_datums.keypoints.keypoints import Keypoints
from sparrow_datums.types import PType


def test_docstring_example():
    result = doctest.testmod(heatmaps)
    assert result.attempted > 0
    assert result.failed == 0


def test_is_heatmap():
    heatmap_array = np.ones((5, 5))
    heatmap = Heatmaps(data=heatmap_array, ptype=PType.heatmap)
    assert heatmap.is_heatmap


def test_back_and_forth_between_keypoints_and_heatmaps_for_1D_arrays():
    keypoint = Keypoints(
        np.array([4, 8]),
        PType.absolute_kp,
        image_height=21,
        image_width=28,
    )
    heatmap_array = keypoint.to_heatmap_array()
    assert heatmap_array.shape == (keypoint.image_height, keypoint.image_width)
    heatmap = Heatmaps(
        heatmap_array,
        PType.heatmap,
        image_width=heatmap_array.shape[-1],
        image_height=heatmap_array.shape[-2],
    )
    assert heatmap.is_heatmap
    assert heatmap.shape == (keypoint.image_height, keypoint.image_width)
    assert np.max(heatmap.array) == 1
    assert np.min(heatmap.array) == 0
    back_to_keypoint = heatmap.to_keypoint()
    assert type(back_to_keypoint) is Keypoints
    assert back_to_keypoint[0][0] == keypoint.array[0]
    assert back_to_keypoint[0][1] == keypoint.array[1]
    back_to_keypoint_abs = back_to_keypoint.to_absolute()
    assert back_to_keypoint_abs.is_absolute
    assert back_to_keypoint_abs[0][0] == keypoint.array[0] * keypoint.image_width
    assert back_to_keypoint_abs[0][1] == keypoint.array[1] * keypoint.image_height
    back_to_keypoint_rel = back_to_keypoint.to_relative()
    assert back_to_keypoint_rel.is_relative


def test_back_and_forth_between_keypoints_and_heatmaps_for_dense_arrays():
    n_keypoints = 11
    np.random.seed(0)
    candidate_arr = np.random.randint(low=2, high=20, size=(n_keypoints, 2))
    keypoint = Keypoints(
        candidate_arr,
        PType.absolute_kp,
        image_height=21,
        image_width=28,
    )
    heatmap_array = keypoint.to_heatmap_array()
    assert heatmap_array.shape == (
        n_keypoints,
        keypoint.image_height,
        keypoint.image_width,
    )
    heatmap = Heatmaps(
        heatmap_array,
        PType.heatmap,
        image_width=heatmap_array.shape[-1],
        image_height=heatmap_array.shape[-2],
    )
    assert heatmap.is_heatmap
    assert heatmap.shape == (n_keypoints, keypoint.image_height, keypoint.image_width)
    assert np.max(heatmap.array) == 1
    assert np.min(heatmap.array) == 0
    back_to_keypoint = heatmap.to_keypoint()
    assert type(back_to_keypoint) is Keypoints
    assert back_to_keypoint[7][0] == keypoint.array[7][0]
    assert back_to_keypoint[7][1] == keypoint.array[7][1]
    back_to_keypoint_abs = back_to_keypoint.to_absolute()
    assert back_to_keypoint_abs.is_absolute
    assert back_to_keypoint_abs[7][0] == keypoint.array[7][0] * keypoint.image_width
    assert back_to_keypoint_abs[7][1] == keypoint.array[7][1] * keypoint.image_height
    back_to_keypoint_rel = back_to_keypoint.to_relative()
    assert back_to_keypoint_rel.is_relative
