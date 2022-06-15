"""Base class for dense data arrays with metadata."""
from __future__ import annotations

import abc
import gzip
import json
from pathlib import Path
from typing import Any, TypeVar

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
    start_time: float, optional
        Start time of chunk with respect to video (if tracking)
    """

    def __new__(
        cls: type[T],
        data: FloatArray,
        ptype: PType = PType.unknown,
        image_width: int | None = None,
        image_height: int | None = None,
        fps: float | None = None,
        object_ids: list[str] | None = None,
        start_time: float | None = None,
    ) -> T:
        """Instantiate a new chunk."""
        obj: T = np.asarray(data).view(cls)
        obj.ptype = ptype
        obj._image_width = image_width
        obj._image_height = image_height
        obj._fps = fps
        obj._object_ids = object_ids
        obj._start_time = start_time
        obj._scale = None
        return obj

    def __array_finalize__(self, obj: T | None) -> None:
        """Instantiate a new chunk created from a view."""
        if obj is None:
            return
        self.ptype: PType = getattr(obj, "ptype", PType.unknown)
        self._image_width: float | None = getattr(obj, "_image_width", None)
        self._image_height: float | None = getattr(obj, "_image_height", None)
        self._fps: float | None = getattr(obj, "_fps", None)
        self._object_ids: list[str] | None = getattr(obj, "_object_ids", None)
        self._start_time: float | None = getattr(obj, "_start_time", None)
        self._scale: FloatArray | None = getattr(obj, "_scale", None)
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
    def fps(self) -> float:
        """Frames per second."""
        if self._fps is None:
            raise ValueError("fps not set")
        return self._fps

    @property
    def object_ids(self) -> list[str]:
        """List of IDs for tracked objects."""
        if self._object_ids is None:
            raise ValueError("object_ids not set")
        return self._object_ids

    @property
    def start_time(self) -> float:
        """Start time of chunk with respect to a video."""
        if self._start_time is None:
            raise ValueError("start_time not set")
        return self._start_time

    @property
    def scale(self) -> FloatArray:
        """Scaling array."""
        if self._scale is None:
            width = self.image_width
            height = self.image_height
            self._scale = np.array([width, height, width, height])
        return self._scale

    @property
    def duration(self) -> float:
        """Duration of the chunk in seconds."""
        raise NotImplementedError

    @property
    def metadata_kwargs(self) -> dict[str, Any]:
        """Metadata kwargs for downstream constructors."""
        return {
            "image_width": self._image_width,
            "image_height": self._image_height,
            "fps": self._fps,
            "object_ids": self._object_ids,
            "start_time": self._start_time,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a dict."""
        return {
            "data": np.where(np.isnan(self.array), np.array(None), self.array).tolist(),
            "classname": self.__class__.__name__,
            "ptype": self.ptype.name,
            **self.metadata_kwargs,
        }

    def to_file(self, path: str | Path) -> None:
        """Write serialized chunk to disk."""
        if not str(path).endswith(".json.gz"):
            raise ValueError("Chunk file name must end with .json.gz")
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(
        cls: type[T], chunk_dict: dict[str, Any], dims: int | None = None
    ) -> T:
        """Create chunk from chunk dict."""
        data: FloatArray
        if len(chunk_dict["data"]) == 0 and dims:
            if "tracking" in cls.__name__.lower():
                data = np.zeros((0, 0, dims), "float64")
            else:
                data = np.zeros((0, dims), "float64")
        else:
            data = np.array(chunk_dict["data"]).astype("float64")
        data[data == None] = np.nan
        return cls(
            data,
            ptype=PType(chunk_dict["ptype"]),
            image_width=chunk_dict["image_width"],
            image_height=chunk_dict["image_height"],
            fps=chunk_dict["fps"],
            object_ids=chunk_dict["object_ids"],
            start_time=chunk_dict.get("start_time"),  # Backwards compatibility
        )

    @classmethod
    def from_file(cls: type[T], path: str | Path) -> T:
        """Read chunk from disk."""
        with gzip.open(path, "rt") as f:
            return cls.from_dict(json.loads(f.read()))
