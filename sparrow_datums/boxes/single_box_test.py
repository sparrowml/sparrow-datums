import numpy as np
import pytest

from .single_box import SingleBox


def test_single_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValueError):
        SingleBox(np.ones((2, 4)))
    with pytest.raises(ValueError):
        SingleBox(np.ones(3))
    # This should work
    SingleBox(np.random.uniform(size=4))
