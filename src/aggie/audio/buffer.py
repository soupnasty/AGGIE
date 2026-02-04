"""Audio ring buffer for pre-roll capture."""

from collections import deque
from typing import Optional

import numpy as np


class AudioRingBuffer:
    """Ring buffer that keeps the last N seconds of audio.

    Used to capture audio before wake word is fully detected,
    ensuring the beginning of the user's utterance isn't lost.
    """

    def __init__(self, duration_seconds: float, sample_rate: int = 16000) -> None:
        """Initialize ring buffer.

        Args:
            duration_seconds: Maximum duration to buffer.
            sample_rate: Audio sample rate in Hz.
        """
        self._max_samples = int(duration_seconds * sample_rate)
        self._buffer: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._sample_rate = sample_rate

    @property
    def duration_seconds(self) -> float:
        """Current buffered duration in seconds."""
        return self._total_samples / self._sample_rate

    @property
    def sample_count(self) -> int:
        """Number of samples currently buffered."""
        return self._total_samples

    def append(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer, evicting old data if needed.

        Args:
            frame: Audio frame to add.
        """
        self._buffer.append(frame)
        self._total_samples += len(frame)

        # Evict old frames if over capacity
        while self._total_samples > self._max_samples and len(self._buffer) > 1:
            removed = self._buffer.popleft()
            self._total_samples -= len(removed)

    def get_all(self) -> np.ndarray:
        """Get all buffered audio as a single array.

        Returns:
            Concatenated audio data, or empty array if buffer is empty.
        """
        if not self._buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(list(self._buffer))

    def get_last(self, duration_seconds: float) -> np.ndarray:
        """Get the last N seconds of buffered audio.

        Args:
            duration_seconds: Duration to retrieve.

        Returns:
            Audio data from the last N seconds.
        """
        if not self._buffer:
            return np.array([], dtype=np.int16)

        all_audio = self.get_all()
        samples_needed = int(duration_seconds * self._sample_rate)

        if len(all_audio) <= samples_needed:
            return all_audio

        return all_audio[-samples_needed:]

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_samples = 0

    def copy_to_list(self) -> list[np.ndarray]:
        """Get a copy of the buffer as a list of frames.

        Returns:
            List of audio frames (copies, safe to modify).
        """
        return [frame.copy() for frame in self._buffer]
