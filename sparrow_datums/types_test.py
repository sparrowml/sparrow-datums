from sparrow_datums.types import BoxType

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
