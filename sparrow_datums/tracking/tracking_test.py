import numpy as np
import pytest

from .tracking import Tracking


def test_validate_tracking_chunk():
    with pytest.raises(ValueError):
        Tracking(np.ones((2, 4)))
    Tracking(np.ones((2, 2, 2)))


def test_resample_random_chunk():
    x = np.random.uniform(size=(10, 5, 3))
    chunk = Tracking(x, fps=20)
    fast_chunk = chunk.resample(40)
    assert fast_chunk.fps == 40
    assert len(fast_chunk) == 20
    slow_chunk = fast_chunk.resample(20)
    np.testing.assert_equal(chunk.array, slow_chunk.array)


def test_valid_duration():
    tracking = Tracking(np.ones((2, 2, 2)), fps=30)
    assert tracking.duration == 1 / 15
