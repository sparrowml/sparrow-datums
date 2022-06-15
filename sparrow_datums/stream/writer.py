from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..chunk import T
from .types import ChunkPath, Footer, Header


class ChunkStreamWriter:
    """A class for writing chunk streams to disk."""

    def __init__(
        self,
        manifest_path: str | Path,
        chunk_type: type[T],
        fps: float,
        start_time: float = 0,
        ptype: Optional[str] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        object_ids: Optional[list[str]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.chunk_type = chunk_type
        self.header = Header(
            classname=chunk_type.__name__,
            ptype=ptype,
            image_width=image_width,
            image_height=image_height,
            fps=fps,
            start_time=start_time,
            object_ids=object_ids,
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
        if not isinstance(chunk, self.chunk_type):
            raise TypeError(
                (f"Incorrect chunk type {type(chunk)}. Expected {self.chunk_type}.")
            )
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
