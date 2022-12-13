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

    @property
    def is_relative(self) -> bool:
        """Whether the parameterization is in relative space."""
        return "relative" in self.name

    @property
    def is_absolute(self) -> bool:
        """Whether the parameterization is in absolute space."""
        return "absolute" in self.name

    @property
    def as_relative(self) -> "PType":
        """Convert parameterization type to relative."""
        return PType(self.name.replace("absolute", "relative"))

    @property
    def as_absolute(self) -> "PType":
        """Convert parameterization type to absolute."""
        return PType(self.name.replace("relative", "absolute"))
