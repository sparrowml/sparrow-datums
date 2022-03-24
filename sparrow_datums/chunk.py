"""Base class for dense data arrays with metadata."""
from __future__ import annotations

import abc
import gzip
import json
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import numpy as np

from .types import FloatArray, PType

T = TypeVar("T", bound="Chunk")


class Chunk(FloatArray):
    """
    Base class for dense data arrays with metadata.

    Parameters
    ----------
    data : FloatArray
        A numpy array of dense floats
    ptype : PType
        The parameterization of the dense data
    image_width : int, optional
        The width of the relevant image
    image_height : int, optional
        The height of the relevant image
    fps : float, optional
        The framerate of the chunk data (if tracking)
    object_ids:  list[str], optional
        Identifiers for the objects (if tracking)
    """

    def __new__(
        cls: type[T],
        data: FloatArray,
        ptype: PType = PType.unknown,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        fps: Optional[float] = None,
        object_ids: Optional[list[str]] = None,
    ) -> T:
        """Instantiate a new chunk."""
        obj: T = np.asarray(data).view(cls)
        obj.ptype = ptype
        obj._image_width = image_width
        obj._image_height = image_height
        obj._fps = fps
        obj._object_ids = object_ids
        obj._scale = None
        return obj

    def __array_finalize__(self, obj: Optional[T]) -> None:
        """Instantiate a new chunk created from a view."""
        if obj is None:
            return
        self.ptype: PType = getattr(obj, "ptype", PType.unknown)
        self._image_width: Optional[float] = getattr(obj, "_image_width", None)
        self._image_height: Optional[float] = getattr(obj, "_image_height", None)
        self._fps: Optional[float] = getattr(obj, "_fps", None)
        self._object_ids: Optional[list[str]] = getattr(obj, "_object_ids", None)
        self._scale: Optional[FloatArray] = getattr(obj, "_scale", None)
        self.validate()

    @abc.abstractmethod
    def validate(self) -> None:
        """Raise ValueError for incorrect shape or values."""
        raise NotImplementedError

    @property
    def array(self) -> FloatArray:
        """Dense data as an ndarray."""
        return self.view(np.ndarray)

    @property
    def image_width(self) -> float:
        """Image width."""
        if self._image_width is None:
            raise ValueError("image_width not set")
        return self._image_width

    @property
    def image_height(self) -> float:
        """Image height."""
        if self._image_height is None:
            raise ValueError("image_height not set")
        return self._image_height

    @property
    def scale(self) -> FloatArray:
        """Scaling array."""
        if self._scale is None:
            width = self.image_width
            height = self.image_height
            self._scale = np.array([width, height, width, height])
        return self._scale

    @property
    def metadata_kwargs(self) -> dict[str, Any]:
        """Metadata kwargs for downstream constructors."""
        return {
            "image_width": self._image_width,
            "image_height": self._image_height,
            "fps": self._fps,
            "object_ids": self._object_ids,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a dict."""
        return {
            "data": np.where(np.isnan(self.array), np.array(None), self.array).tolist(),
            "classname": self.__class__.__name__,
            "ptype": self.ptype.name,
            **self.metadata_kwargs,
        }

    def to_file(self, path: Union[str, Path]) -> None:
        """Write serialized chunk to disk."""
        if not str(path).endswith(".json.gz"):
            raise ValueError("Chunk file name must end with .json.gz")
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls: type[T], chunk_dict: dict[str, Any]) -> T:
        """Create chunk from chunk dict."""
        data: FloatArray = np.array(chunk_dict["data"]).astype("float64")
        data[data == None] = np.nan
        return cls(
            data,
            ptype=PType(chunk_dict["ptype"]),
            image_width=chunk_dict["image_width"],
            image_height=chunk_dict["image_height"],
            fps=chunk_dict["fps"],
            object_ids=chunk_dict["object_ids"],
        )

    @classmethod
    def from_file(cls: type[T], path: Union[str, Path]) -> T:
        """Read chunk from disk."""
        with gzip.open(path, "rt") as f:
            return cls.from_dict(json.loads(f.read()))
