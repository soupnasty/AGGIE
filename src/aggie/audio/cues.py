"""Audio feedback cues for state transitions.

Generates simple tones programmatically - no external audio files needed.
"""

import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CueType(Enum):
    """Types of audio cues."""

    WAKE = auto()  # Wake word detected - start listening
    DONE_LISTENING = auto()  # Finished recording - processing
    ERROR = auto()  # Something went wrong


class AudioCues:
    """Generates and manages audio feedback cues.

    Creates simple sine wave tones that play at state transitions
    to give users audible feedback.
    """

    SAMPLE_RATE = 22050  # Standard sample rate for cues

    # Tone definitions: (frequencies_hz, durations_ms, volumes)
    CUE_DEFINITIONS = {
        # Wake: ascending two-tone chime (C5 -> E5)
        CueType.WAKE: {
            "frequencies": [523, 659],  # C5, E5
            "durations": [80, 120],
            "volume": 0.3,
        },
        # Done listening: single short beep (G5)
        CueType.DONE_LISTENING: {
            "frequencies": [784],  # G5
            "durations": [100],
            "volume": 0.25,
        },
        # Error: descending two-tone (E5 -> C5)
        CueType.ERROR: {
            "frequencies": [659, 523],  # E5, C5
            "durations": [100, 150],
            "volume": 0.3,
        },
    }

    def __init__(self) -> None:
        """Initialize audio cues generator."""
        self._cache: dict[CueType, np.ndarray] = {}
        self._generate_all_cues()

    def _generate_all_cues(self) -> None:
        """Pre-generate all cue audio data."""
        for cue_type, definition in self.CUE_DEFINITIONS.items():
            self._cache[cue_type] = self._generate_cue(
                frequencies=definition["frequencies"],
                durations=definition["durations"],
                volume=definition["volume"],
            )
        logger.debug(f"Generated {len(self._cache)} audio cues")

    def _generate_cue(
        self,
        frequencies: list[int],
        durations: list[int],
        volume: float,
    ) -> np.ndarray:
        """Generate a multi-tone cue.

        Args:
            frequencies: List of frequencies in Hz for each tone.
            durations: List of durations in ms for each tone.
            volume: Volume multiplier (0.0 - 1.0).

        Returns:
            Audio data as int16 numpy array.
        """
        segments = []

        for freq, duration_ms in zip(frequencies, durations):
            num_samples = int(self.SAMPLE_RATE * duration_ms / 1000)
            t = np.linspace(0, duration_ms / 1000, num_samples, dtype=np.float32)

            # Generate sine wave
            tone = np.sin(2 * np.pi * freq * t)

            # Apply envelope to avoid clicks (fade in/out)
            envelope = self._create_envelope(num_samples)
            tone = tone * envelope * volume

            segments.append(tone)

            # Small gap between tones
            if len(frequencies) > 1:
                gap = np.zeros(int(self.SAMPLE_RATE * 0.02), dtype=np.float32)
                segments.append(gap)

        # Concatenate and convert to int16
        audio = np.concatenate(segments)
        audio_int16 = (audio * 32767).astype(np.int16)

        return audio_int16

    def _create_envelope(self, num_samples: int) -> np.ndarray:
        """Create a smooth envelope to avoid audio clicks.

        Args:
            num_samples: Total number of samples.

        Returns:
            Envelope array (0.0 - 1.0).
        """
        # 5ms fade in/out
        fade_samples = min(int(self.SAMPLE_RATE * 0.005), num_samples // 4)

        envelope = np.ones(num_samples, dtype=np.float32)

        # Fade in
        if fade_samples > 0:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        return envelope

    def get_cue(self, cue_type: CueType) -> tuple[np.ndarray, int]:
        """Get audio data for a cue type.

        Args:
            cue_type: Type of cue to get.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        if cue_type not in self._cache:
            logger.warning(f"Unknown cue type: {cue_type}")
            return np.array([], dtype=np.int16), self.SAMPLE_RATE

        return self._cache[cue_type], self.SAMPLE_RATE
