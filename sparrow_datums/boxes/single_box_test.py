import numpy as np
import pytest

from ..chunk_types import PType
from ..exceptions import ValidationError
from .single_box import SingleBox


def test_single_box_conversion_creates_single_box():
    box_a = SingleBox(np.ones(4), PType.absolute_tlbr)
    box_b = box_a.to_tlwh()
    assert box_b.is_tlwh
    assert isinstance(box_b, SingleBox)


def test_single_box_with_bad_shape_throws_value_error():
    with pytest.raises(ValidationError):
        SingleBox(np.ones((2, 4)))
    with pytest.raises(ValidationError):
        SingleBox(np.ones(3))
