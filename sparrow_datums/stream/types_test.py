import json

from sparrow_datums import FrameAugmentedBoxes, PType

from .types import ChunkPath, Header


def test_header_serialization():
    value_str = Header(
        classname=FrameAugmentedBoxes.__name__,
        ptype=PType.absolute_tlbr.name,
        image_width=None,
        image_height=None,
        fps=1,
        start_time=0,
    ).to_json()
    value = json.loads(value_str)
    assert value["classname"] == "FrameAugmentedBoxes"
    assert value["image_height"] is None


def test_chunk_path_serialization():
    value_dict = {"path": "0000.json.gz", "duration": 4.0, "start_time": 0}
    value = ChunkPath.from_dict(value_dict)
    assert value.duration == 4.0
