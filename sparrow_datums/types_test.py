from .types import BoxType

BOX_TYPE_NAMES = (
    "relative_tlbr",
    "relative_tlwh",
    "absolute_tlbr",
    "absolute_tlwh",
)


def test_to_relative_creates_relative_boxes():
    for type_name in BOX_TYPE_NAMES:
        box_type = BoxType(type_name)
        assert box_type.as_relative.is_relative


def test_to_absolute_creates_relative_boxes():
    for type_name in BOX_TYPE_NAMES:
        box_type = BoxType(type_name)
        assert box_type.as_absolute.is_absolute


def test_to_tlbr_creates_tlbr_boxes():
    for type_name in BOX_TYPE_NAMES:
        box_type = BoxType(type_name)
        assert box_type.as_tlbr.is_tlbr


def test_to_tlwh_creates_tlwh_boxes():
    for type_name in BOX_TYPE_NAMES:
        box_type = BoxType(type_name)
        assert box_type.as_tlwh.is_tlwh
