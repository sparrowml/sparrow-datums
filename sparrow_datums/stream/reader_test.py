import os
import tempfile

import numpy as np

from sparrow_datums import BoxTracking

from .reader import ChunkStreamReader
from .writer import ChunkStreamWriter


def test_chunk_stream_reader():
    with tempfile.TemporaryDirectory() as dir:
        manifest_path = os.path.join(dir, "stream.jsonl")
        writer = ChunkStreamWriter(manifest_path, BoxTracking, fps=1)
        a = BoxTracking(np.ones((2, 2, 4)), fps=1, start_time=0)
        b = BoxTracking(np.ones((2, 2, 4)), fps=1, start_time=a.duration)
        c = BoxTracking(np.ones((2, 1, 4)), fps=1, start_time=a.duration + b.duration)
        writer.add_chunk(a)
        writer.add_chunk(b)
        writer.add_chunk(c)
        writer.close()
        reader = ChunkStreamReader(manifest_path, BoxTracking)
        assert reader.header.fps == 1
        assert len(reader) == 3
        assert isinstance(reader[1], BoxTracking)
        assert all(isinstance(chunk, BoxTracking) for chunk in reader)
        assert reader.concat().shape == (6, 2, 4)


def test_chunk_stream_reader_concat_with_object_ids():
    with tempfile.TemporaryDirectory() as dir:
        manifest_path = os.path.join(dir, "stream.jsonl")
        writer = ChunkStreamWriter(manifest_path, BoxTracking, fps=1)
        a = BoxTracking(np.ones((2, 2, 4)), fps=1, start_time=0, object_ids=["1", "2"])
        b = BoxTracking(
            np.ones((2, 2, 4)), fps=1, start_time=a.duration, object_ids=["2", "3"]
        )
        writer.add_chunk(a)
        writer.add_chunk(b)
        writer.close()
        reader = ChunkStreamReader(manifest_path, BoxTracking)
        assert reader.concat().shape == (4, 3, 4)
