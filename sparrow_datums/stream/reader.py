from __future__ import annotations

from pathlib import Path

from ..chunk import T
from .types import ChunkPath, Footer, Header


class ChunkStreamReader:
    """A class for reading chunk streams from disk."""

    def __init__(self, manifest_path: str | Path, chunk_type: type[T]) -> None:
        self.manifest_path = Path(manifest_path)
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
        chunk_path = self.manifest_path.parent / self.chunk_paths[idx].path
        return self.chunk_type.from_file(chunk_path)
