from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..chunk import T
from ..chunk_types import PType
from ..exceptions import ValidationError
from ..tracking import Tracking
from .stream_types import ChunkPath, Footer, Header


class ChunkStreamReader:
    """A class for reading chunk streams from disk."""

    def __init__(self, manifest_path: str | Path, chunk_type: type[T]) -> None:
        self.manifest_path = Path(manifest_path)
        if not issubclass(chunk_type, Tracking):
            raise ValidationError(f"Invalid stream chunk type {chunk_type}")
        self.chunk_type = chunk_type
        header, chunk_paths, footer = self.parse_manifest()
        self.header = header
        self.chunk_paths = chunk_paths
        self.footer = footer

    def parse_manifest(self) -> tuple[Header, list[ChunkPath], Footer]:
        """Write the manifest with all the chunks."""
        with open(self.manifest_path) as f:
            value_strings = f.read().split("\n")
        if len(value_strings) < 2:
            raise ValueError(f"Invalid manifest at {self.manifest_path}")
        header = Header.from_json(value_strings[0])
        footer = Footer.from_json(value_strings[-1])
        chunk_paths = list(map(ChunkPath.from_json, value_strings[1:-1]))
        return header, chunk_paths, footer

    def __len__(self) -> int:
        """Get length of chunk stream."""
        return len(self.chunk_paths)

    def __getitem__(self, idx: int) -> T:
        """Get a chunk from a stream."""
        chunk_path_object = self.chunk_paths[idx]
        chunk_path = self.manifest_path.parent / chunk_path_object.path
        try:
            return self.chunk_type.from_file(chunk_path)
        except ValidationError:
            chunk = self.chunk_type.empty(
                ptype=PType[self.header.ptype],
                image_width=self.header.image_width,
                image_height=self.header.image_height,
                fps=self.header.fps,
                start_time=chunk_path_object.start_time,
            )
            new_shape = list(chunk.shape)
            new_shape[0] = round(chunk_path_object.duration * self.header.fps)
            return chunk.pad(tuple(new_shape))

    def __iter__(self) -> Iterator[T]:
        """Yield chunks from a stream."""
        for i in range(len(self)):
            yield self[i]

    def concat(self) -> T:
        n_frames = 0
        object_ids = set()
        n_objects = 0
        for chunk in self:
            n_frames += len(chunk)
            n_objects = max(n_objects, chunk.shape[1])
            object_ids |= set(chunk.object_ids)
        object_ids = sorted(object_ids)
        n_objects = max(n_objects, len(object_ids))
        data = np.zeros((n_frames, n_objects, chunk.shape[-1])) * np.nan
        frame_idx = 0
        for chunk in self:
            object_indices = np.array(
                [object_ids.index(object_id) for object_id in chunk.object_ids]
            )
            if len(object_indices) < chunk.shape[1]:
                object_indices = np.arange(chunk.shape[1])
            if len(object_indices):
                data[frame_idx : frame_idx + len(chunk), object_indices] = chunk.data
            frame_idx += len(chunk)
        return self.chunk_type(
            data,
            ptype=chunk.ptype,
            **chunk.metadata_kwargs,
        )
