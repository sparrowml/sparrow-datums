import abc
import enum
import gzip
import json
from typing import Any, Dict, Optional

import numpy as np


class Chunk(np.ndarray):
    """
    Base class for NumPy array sublcasses

    Parameters
    ----------
    data : np.ndarray
        A dense ndarray of data
    type : enum.Enum
        The parameterization of the data array
    image_width : float, optional
        The width of the image
    image_height : float, optional
        The height of the image
    """

    def __new__(
        cls,
        data: np.ndarray,
        type: Optional[enum.Enum] = None,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> None:
        cls._type = type
        cls._image_width = image_width
        cls._image_height = image_height
        cls._scale = None
        return super().__new__(cls, data.shape, dtype=data.dtype, buffer=data.data)

    def __init__(self, *args, **kwargs) -> None:
        """ndarray subclasses don't need __init__, but pylance does"""
        pass

    def __array_finalize__(self, _) -> None:
        self._check_shape()

    @abc.abstractmethod
    def _check_shape(self) -> None:
        """Raise ValueError for incorrect shape"""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def decode_type(cls, type_name: str) -> enum.Enum:
        """Decode the type string"""
        raise NotImplementedError

    @property
    def array(self) -> np.ndarray:
        """Dense data as an ndarray"""
        return self.view(np.ndarray)

    @property
    def image_width(self) -> float:
        """Image width"""
        if self._image_width is None:
            raise ValueError("image_width not set")
        return self._image_width

    @property
    def image_height(self) -> float:
        """Image height"""
        if self._image_height is None:
            raise ValueError("image_height not set")
        return self._image_height

    @property
    def scale(self) -> np.ndarray:
        """Scaling array"""
        if self._scale is None:
            width = self.image_width
            height = self.image_height
            self._scale = np.array([width, height, width, height])
        return self._scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": np.where(np.isnan(self), None, self).tolist(),
            "classname": self.__class__.__name__,
            "type": self._type.name if self._type else None,
            "image_width": self._image_width,
            "image_height": self._image_height,
        }

    def to_file(self, path: str) -> None:
        if not path.endswith(".json.gz"):
            raise ValueError("Chunk file name must end with .json.gz")
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, chunk_dict: Dict[str, Any]) -> "Chunk":
        classname = chunk_dict["classname"]
        if classname != cls.__name__:
            raise TypeError(
                f"Expected classname of {cls.__name__} but found {classname}"
            )
        data = np.array(chunk_dict["data"])
        data[data == None] = np.nan
        return cls(
            data.astype("float64"),
            type=cls.decode_type(chunk_dict["type"]) if chunk_dict["type"] else None,
            image_width=chunk_dict["image_width"],
            image_height=chunk_dict["image_height"],
        )

    @classmethod
    def from_file(cls, path: str) -> "Chunk":
        with gzip.open(path, "rt") as f:
            return cls.from_dict(json.loads(f.read()))
