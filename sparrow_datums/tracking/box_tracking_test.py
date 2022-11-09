import numpy as np
import pytest

from ..exceptions import ValidationError
from .box_tracking import BoxTracking


def test_box_tracking_chunk_requires_3_dimensions():
    with pytest.raises(ValidationError):
        BoxTracking(np.ones((2, 4)))
    with pytest.raises(ValidationError):
        BoxTracking(np.ones((2, 2, 2)))
