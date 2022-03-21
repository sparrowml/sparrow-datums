import enum


class PType(enum.Enum):
    """
    Enum for chunk parameterization types
    """

    unknown = "unknown"
    """
    Unknown parameterization
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
        return "relative" in self.name

    @property
    def is_absolute(self) -> bool:
        return "absolute" in self.name

    @property
    def is_tlbr(self) -> bool:
        return "tlbr" in self.name

    @property
    def is_tlwh(self) -> bool:
        return "tlwh" in self.name

    @property
    def as_relative(self) -> "PType":
        """Convert box type to"""
        return PType(self.name.replace("absolute", "relative"))

    @property
    def as_absolute(self) -> "PType":
        return PType(self.name.replace("relative", "absolute"))

    @property
    def as_tlbr(self) -> "PType":
        return PType(self.name.replace("tlwh", "tlbr"))

    @property
    def as_tlwh(self) -> "PType":
        return PType(self.name.replace("tlbr", "tlwh"))
