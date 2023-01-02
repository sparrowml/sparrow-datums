from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, TypeVar, Union

T = TypeVar("T", bound="StreamValue")
ValueDict = Dict[str, Union[str, int, float, bool]]


@dataclass
class StreamValue:
    """Base class for JSON lines values in manifest file."""

    def to_dict(self) -> ValueDict:
        """Serialize the value as a dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the value as a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls: type[T], value_dict: ValueDict) -> T:
        """Create a StreamValue object from a dict."""
        return cls(**value_dict)

    @classmethod
    def from_json(cls: type[T], value_str: str) -> T:
        """Create a StreamValue object from a JSON string."""
        return cls.from_dict(json.loads(value_str))


@dataclass
class Header(StreamValue):
    """
    Header information for a chunk stream manifest.

    This is roughly equivalent to chunk metadata.
    """

    classname: str
    ptype: Union[str, None]
    image_width: Union[int, None]
    image_height: Union[int, None]
    fps: float
    start_time: float


@dataclass
class ChunkPath(StreamValue):
    """Information about a chunk in the stream for the manifest."""

    path: str
    start_time: float
    duration: float


@dataclass
class Footer(StreamValue):
    """Footer information for a chunk stream manifest."""

    is_done: bool
