from sparrow_datums.boxes import BoxType


def test_to_relative_creates_relative_boxes():
    for i in range(4):
        box_type = BoxType(i)
        assert box_type.as_relative.is_relative


def test_to_absolute_creates_relative_boxes():
    for i in range(4):
        box_type = BoxType(i)
        assert box_type.as_absolute.is_absolute
