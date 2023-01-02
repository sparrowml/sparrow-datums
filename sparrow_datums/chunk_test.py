import os
import tempfile

import numpy as np
import pytest

from .chunk import Chunk
from .chunk_types import PType
from .exceptions import ValidationError


class Polygons(Chunk):
    empty_shape: tuple[int] = (2,)

    def validate(self) -> None:
        if self.shape[-1] != 2:
            raise ValidationError("Uh oh")


def test_dense_serialization_preserves_data():
    polygons_a = Polygons(np.random.uniform(size=(2,)), ptype=PType.unknown)
    polygons_a_dict = polygons_a.to_dict()
    # Classname gets serialized
    assert polygons_a_dict["classname"] == "Polygons"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.json.gz")
        polygons_a.to_file(path)
        polygons_b = Polygons.from_file(path)
    # Serialization preserves dense data
    np.testing.assert_allclose(polygons_a.array, polygons_b.array)
    # Chunk type gets pulled back in
    assert polygons_b.ptype == PType.unknown


def test_nan_in_dense_gets_preserved():
    x = np.ones((10, 2))
    x[:, 0] *= np.nan
    polygons_a = Polygons(x)
    polygons_a_dict = polygons_a.to_dict()
    polygons_b = Polygons.from_dict(polygons_a_dict)
    np.testing.assert_allclose(polygons_a.array, polygons_b.array)
    assert np.isnan(polygons_a.array).mean() == np.isnan(polygons_b.array).mean()


def test_chunk_is_an_ndarray():
    polygons = Polygons(np.ones(2))
    assert isinstance(polygons, np.ndarray)


def test_duration_raises_by_default():
    polygons = Polygons(np.ones(2))
    with pytest.raises(NotImplementedError):
        polygons.duration


def test_pad_adds_nan_values():
    polygons_a = Polygons(np.random.uniform(size=(2, 2)), ptype=PType.unknown)
    polygons_b = polygons_a.pad((4, 2))
    assert polygons_b.shape == (4, 2)
    assert np.isnan(polygons_b.array).mean() == 0.5


def test_empty_polygons():
    empty_polygons = Polygons.empty()
    assert empty_polygons.shape == (2,)
