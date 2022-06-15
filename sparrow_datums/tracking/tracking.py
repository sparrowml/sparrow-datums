from typing import TypeVar

import numpy as np

from ..chunk import Chunk

T = TypeVar("T", bound="Tracking")


class Tracking(Chunk):
    """Validate that the chunk has 3 dimensions."""

    def validate(self) -> None:
        """Check shape of tracking array."""
        if self.ndim != 3:
            raise ValueError("Tracking chunks must have 3 dimensions")

    @property
    def duration(self) -> float:
        """Compute duration (in seconds) of tracking chunk."""
        return len(self) / self.fps

    def resample(self: T, new_fps: float) -> T:
        """Modify fps for a tracking chunk."""
        n_current_frames = len(self)
        duration_seconds = n_current_frames / self.fps
        n_new_frames = round(duration_seconds * new_fps)
        frame_shape = self.shape[1:]
        data = np.zeros((n_new_frames,) + frame_shape) * np.nan
        for new_idx in range(n_new_frames):
            current_idx = int((new_idx / new_fps) * self.fps)
            data[new_idx] = self.array[current_idx]
        metadata_kwargs = self.metadata_kwargs.copy()
        metadata_kwargs["fps"] = new_fps
        return self.__class__(
            data=data,
            ptype=self.ptype,
            **metadata_kwargs,
        )
