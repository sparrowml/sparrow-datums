from .boxes import Boxes


class SingleBox(Boxes):
    def validate(self) -> None:
        super().validate()
        if self.ndim > 1:
            raise ValueError("Single box must be a 1D array")
