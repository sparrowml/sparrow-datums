from __future__ import annotations

from pathlib import Path


class ChunkStreamWriter:
    """A class for writing chunk streams to disk."""

    def __init__(
        self,
        manifest_path: str | Path,
        fps: float,
        start_time: float = 0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.fps = fps
        self.start_time = start_time
