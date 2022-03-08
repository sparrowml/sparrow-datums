import enum


class BoxType(enum.IntEnum):
    """
    Enum for box parameterizations
    """

    XYXY_ABS = 0
    """
    (x1, y1, x2, y2) in absolute pixel coordinates
    """

    XYWH_ABS = 1
    """
    (x, y, w, h) in absolute pixel coordinates
    """

    XYXY_REL = 2
    """
    (x1, y1, x2, y2) in relative pixel coordinates
    """

    XYWH_REL = 3
    """
    (x, y, w, h) in relative pixel coordinates
    """

    @property
    def is_relative(self) -> bool:
        return self.name in ("XYXY_REL", "XYWH_REL")

    @property
    def is_absolute(self) -> bool:
        return self.name in ("XYXY_ABS", "XYWH_ABS")

    @property
    def as_relative(self) -> "BoxType":
        """Convert box type to"""
        if self.name == "XYXY_ABS":
            return BoxType.XYXY_REL
        elif self.name == "XYWH_ABS":
            return BoxType.XYWH_REL
        return self

    @property
    def as_absolute(self) -> "BoxType":
        if self.name == "XYXY_REL":
            return BoxType.XYXY_ABS
        elif self.name == "XYWH_REL":
            return BoxType.XYWH_ABS
        return self
