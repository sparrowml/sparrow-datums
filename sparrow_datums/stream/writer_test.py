import json
import os
import tempfile

import numpy as np
import pytest

from sparrow_datums import BoxTracking

from .writer import ChunkStreamWriter


def test_constructor_writes_manifest_file():
    with tempfile.TemporaryDirectory() as dir:
        manifest_path = os.path.join(dir, "stream.jsonl")
        writer = ChunkStreamWriter(manifest_path, BoxTracking, fps=1)
        with open(manifest_path) as f:
            manifest_values = f.read().split("\n")
        assert len(manifest_values) == 2
        assert writer.footer.is_done == False
        writer.close()
        assert writer.footer.is_done


def test_writer_writes_chunks():
    with tempfile.TemporaryDirectory() as dir:
        manifest_path = os.path.join(dir, "stream.jsonl")
        writer = ChunkStreamWriter(manifest_path, BoxTracking, fps=1)
        a = BoxTracking(np.ones((2, 2, 4)), fps=1, start_time=0)
        b = BoxTracking(np.ones((2, 2, 4)), fps=1, start_time=a.duration)
        writer.add_chunk(a)
        writer.add_chunk(b)
        writer.close()
        with open(manifest_path) as f:
            manifest_values = f.read().split("\n")
        assert len(manifest_values) == 4
        for chunk_path in manifest_values[1:-1]:
            data = json.loads(chunk_path)
            _ = BoxTracking.from_file(os.path.join(dir, data["path"]))
        assert writer.footer.is_done


def test_context_manager_closes_stream():
    with pytest.raises(ValueError):
        with tempfile.TemporaryDirectory() as dir:
            manifest_path = os.path.join(dir, "stream.jsonl")
            with ChunkStreamWriter(manifest_path, BoxTracking, fps=1) as writer:
                assert writer.footer.is_done == False
                raise ValueError("foo bar")
    assert writer.footer.is_done


def test_first_chunk_can_set_header():
    with tempfile.TemporaryDirectory() as dir:
        manifest_path = os.path.join(dir, "stream.jsonl")
        writer = ChunkStreamWriter(manifest_path, BoxTracking)
        a = BoxTracking(np.ones((2, 2, 4)), fps=1, start_time=0)
        writer.add_chunk(a)
        assert writer.header.fps == 1
