"""Data types."""
import enum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class PType(enum.Enum):
    """Enum for chunk parameterization types."""

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
        """Whether the parameterization is in relative space."""
        return "relative" in self.name

    @property
    def is_absolute(self) -> bool:
        """Whether the parameterization is in absolute space."""
        return "absolute" in self.name

    @property
    def is_tlbr(self) -> bool:
        """Whether the parameterization is TLBR."""
        return "tlbr" in self.name

    @property
    def is_tlwh(self) -> bool:
        """Whether the parameterization is TLWH."""
        return "tlwh" in self.name

    @property
    def as_relative(self) -> "PType":
        """Convert parameterization type to relative."""
        return PType(self.name.replace("absolute", "relative"))

    @property
    def as_absolute(self) -> "PType":
        """Convert parameterization type to absolute."""
        return PType(self.name.replace("relative", "absolute"))

    @property
    def as_tlbr(self) -> "PType":
        """Convert parameterization type to TLBR."""
        return PType(self.name.replace("tlwh", "tlbr"))

    @property
    def as_tlwh(self) -> "PType":
        """Convert parameterization type to TLWH."""
        return PType(self.name.replace("tlbr", "tlwh"))
