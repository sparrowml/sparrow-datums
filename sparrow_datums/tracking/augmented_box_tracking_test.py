import numpy as np
import pytest

from .augmented_box_tracking import AugmentedBoxTracking


def test_box_tracking_chunk_requires_3_dimensions():
    with pytest.raises(ValueError):
        AugmentedBoxTracking(np.ones((2, 6)))
    with pytest.raises(ValueError):
        AugmentedBoxTracking(np.ones((2, 2, 4)))
