# from typing import Optional

# import enum
# import os
# import tempfile

# import numpy as np
# import pytest

# from .chunk import Chunk
# from .types import BoxType


# class PolygonType(enum.Enum):
#     cool = "cool"
#     not_cool = "not_cool"


# class Polygons(Chunk):
#     _type: PolygonType

#     def validate(self) -> None:
#         if self.shape[-1] != 2:
#             raise ValueError("Uh oh")

#     @property
#     def type(self) -> PolygonType:
#         if self._type is None:
#             raise ValueError("Foo bar")
#         _type: PolygonType = self._type
#         return _type

#     @classmethod
#     def decode_type(cls, type_name: Optional[str]) -> Optional[PolygonType]:
#         if type_name is None:
#             return type_name
#         return PolygonType(type_name)


# def test_dense_with_no_width_throws_on_scale():
#     polygons = Polygons(np.ones((10, 10, 2)))
#     with pytest.raises(ValueError):
#         polygons.scale


# def test_dense_sliciing_by_rows_creates_new_object():
#     polygons = Polygons(np.random.uniform(size=(4, 4, 2)), image_width=100)
#     assert isinstance(polygons[:2], Polygons)
#     assert polygons[:2].image_width == 100


# def test_dense_serialization_preserves_data():
#     polygons_a = Polygons(np.random.uniform(size=(2,)), type=PolygonType.cool)
#     polygons_a_dict = polygons_a.to_dict()
#     # Classname gets serialized
#     assert polygons_a_dict["classname"] == "Polygons"
#     with tempfile.TemporaryDirectory() as tmpdir:
#         path = os.path.join(tmpdir, "test.json.gz")
#         polygons_a.to_file(path)
#         polygons_b = Polygons.from_file(path)
#     # Serialization preserves dense data
#     np.testing.assert_allclose(polygons_a.array, polygons_b.array)
#     # Chunk type gets pulled back in
#     assert polygons_b.type == PolygonType.cool


# def test_nan_in_dense_gets_preserved():
#     x = np.ones((10, 2))
#     x[:, 0] *= np.nan
#     polygons_a = Polygons(x)
#     polygons_a_dict = polygons_a.to_dict()
#     polygons_b = Polygons.from_dict(polygons_a_dict)
#     np.testing.assert_allclose(polygons_a.array, polygons_b.array)
#     assert np.isnan(polygons_a.array).mean() == np.isnan(polygons_b.array).mean()


# def test_chunk_is_an_ndarray():
#     polygons = Polygons(np.ones(2))
#     assert isinstance(polygons, np.ndarray)
