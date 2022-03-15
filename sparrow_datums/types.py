import enum


class BoxType(enum.Enum):
    """
    Enum for box parameterizations
    """

    absolute_tlbr = "absolute_tlbr"
    """
    (x1, y1, x2, y2) in absolute pixel coordinates
    """

    absolute_tlwh = "absolute_tlwh"
    """
    (x, y, w, h) in absolute pixel coordinates
    """

    relative_tlbr = "relative_tlbr"
    """
    (x1, y1, x2, y2) in relative pixel coordinates
    """

    relative_tlwh = "relative_tlwh"
    """
    (x, y, w, h) in relative pixel coordinates
    """

    @property
    def is_relative(self) -> bool:
        return self.name in ("relative_tlbr", "relative_tlwh")

    @property
    def is_absolute(self) -> bool:
        return self.name in ("absolute_tlbr", "absolute_tlwh")

    @property
    def as_relative(self) -> "BoxType":
        """Convert box type to"""
        if self.name == "absolute_tlbr":
            return BoxType.relative_tlbr
        elif self.name == "absolute_tlwh":
            return BoxType.relative_tlwh
        return self

    @property
    def as_absolute(self) -> "BoxType":
        if self.name == "relative_tlbr":
            return BoxType.absolute_tlbr
        elif self.name == "relative_tlwh":
            return BoxType.absolute_tlwh
        return self
