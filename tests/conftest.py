"""Pytest configuration and fixtures for AGGIE tests."""

import asyncio
from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Generate 1 second of test audio (16kHz, mono, int16).

    Returns:
        Tuple of (audio_data, sample_rate).
    """
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440 Hz sine wave at ~50% amplitude
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return audio, sample_rate


@pytest.fixture
def silence_audio() -> tuple[np.ndarray, int]:
    """Generate 1 second of silence (16kHz, mono, int16).

    Returns:
        Tuple of (audio_data, sample_rate).
    """
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.int16)
    return audio, sample_rate


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create temporary config file.

    Returns:
        Path to config file.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  silence_duration: 0.5
  max_recording_duration: 5.0
wakeword:
  threshold: 0.3
logging:
  level: DEBUG
"""
    )
    return config_path
