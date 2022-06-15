from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..chunk import T
from ..types import PType
from .types import ChunkPath, Footer, Header


class ChunkStreamWriter:
    """A class for writing chunk streams to disk."""

    def __init__(
        self,
        manifest_path: str | Path,
        chunk_type: type[T],
        fps: Optional[float] = None,
        start_time: float = 0,
        ptype: Optional[PType] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.chunk_type = chunk_type
        self.header = Header(
            classname=chunk_type.__name__,
            ptype=ptype.name if ptype else None,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            start_time=start_time,
        )
        self.chunk_paths: list[ChunkPath] = []
        self.footer = Footer(is_done=False)
        self.next_start_time = start_time
        self.write_manifest()

    def __enter__(self) -> "ChunkStreamWriter":
        """Enter context manager."""
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Exit context manager."""
        self.close()

    def add_chunk(self, chunk: T) -> None:
        """Add chunk to stream and re-write the manifest."""
        # Fill missing header values
        if self.header.ptype is None:
            self.header.ptype = chunk.ptype.name
        if self.header.image_width is None:
            self.header.image_width = chunk._image_width
        if self.header.image_height is None:
            self.header.image_height = chunk._image_height
        if self.header.fps is None:
            self.header.fps = chunk._fps
        # Validation
        if not isinstance(chunk, self.chunk_type):
            raise TypeError(
                (f"Incorrect chunk type {type(chunk)}. Expected {self.chunk_type}.")
            )
        if self.header.ptype != chunk.ptype.name:
            raise ValueError(
                f"Incorrect PType {chunk.ptype.name}. Expected {self.header.ptype}."
            )
        if self.header.image_width != chunk._image_width:
            raise ValueError(
                f"Incorrect image_width {chunk._image_width}. Expected {self.header.image_width}."
            )
        if self.header.image_height != chunk._image_height:
            raise ValueError(
                f"Incorrect image_height {chunk._image_height}. Expected {self.header.image_height}."
            )
        if self.header.fps != chunk._fps:
            raise ValueError(f"Incorrect fps {chunk._fps}. Expected {self.header.fps}.")
        if chunk.start_time != self.next_start_time:
            raise ValueError(
                f"Incorrect start time {chunk.start_time}. Expected {self.next_start_time}."
            )
        chunk_index = len(self.chunk_paths)
        path = f"{chunk_index:04d}.json.gz"
        # If duration/start_time is missing,
        # we want to fail before we write the chunk to disk.
        duration = chunk.duration
        start_time = chunk.start_time
        chunk.to_file(self.manifest_path.parent / path)
        self.chunk_paths.append(
            ChunkPath(path=path, start_time=start_time, duration=duration)
        )
        self.write_manifest()
        self.next_start_time += duration

    def write_manifest(self) -> None:
        """Write the manifest with all the chunks."""
        values = (
            [self.header.to_json()]
            + [p.to_json() for p in self.chunk_paths]
            + [self.footer.to_json()]
        )
        with open(self.manifest_path, "w") as f:
            f.write("\n".join(values))

    def close(self) -> None:
        """Close the chunk stream."""
        self.footer = Footer(is_done=True)
        self.write_manifest()
