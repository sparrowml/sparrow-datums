import numpy as np
import pytest

from .box_tracking import BoxTracking


def test_box_tracking_chunk_requires_3_dimensions():
    with pytest.raises(ValueError):
        BoxTracking(np.ones((2, 4)))
    with pytest.raises(ValueError):
        BoxTracking(np.ones((2, 2, 2)))
