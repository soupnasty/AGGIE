"""Audio capture module using sounddevice."""

import asyncio
import logging
from typing import AsyncGenerator, Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Type for frame callback
FrameCallback = Callable[[np.ndarray], None]


class AudioCapture:
    """Captures audio from the default input device.

    Uses sounddevice callbacks with direct processing for low latency.
    """

    SAMPLE_RATE = 16000  # Required by openWakeWord and faster-whisper
    CHANNELS = 1
    DTYPE = np.int16
    FRAME_DURATION_MS = 80  # Optimal for openWakeWord (80ms frames)

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
        """Initialize audio capture.

        Args:
            sample_rate: Sample rate in Hz.
            channels: Number of audio channels.
        """
        self._sample_rate = sample_rate
        self._channels = channels
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._frame_callback: Optional[FrameCallback] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Queue for async iteration (fallback mode)
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)

    def set_frame_callback(self, callback: Optional[FrameCallback]) -> None:
        """Set callback for direct frame processing (low latency mode).

        Args:
            callback: Function called with each audio frame, or None to disable.
        """
        self._frame_callback = callback

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Audio callback - called from audio thread, must be fast."""
        if status:
            if status.input_overflow:
                logger.warning("Audio input overflow")
            if status.input_underflow:
                logger.warning("Audio input underflow")

        if not self._running:
            return

        # Extract mono channel and ensure int16
        if indata.ndim > 1:
            frame = indata[:, 0].astype(np.int16)
        else:
            frame = indata.astype(np.int16)

        # Direct callback mode (low latency) - preferred for wake word detection
        if self._frame_callback is not None:
            try:
                self._frame_callback(frame)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")
            # Skip queue when using direct callback
            return

        # Queue for async iteration (fallback mode only)
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._queue.put_nowait,
                    frame.copy(),
                )
            except asyncio.QueueFull:
                pass  # Drop frame if queue full

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        """Frame size in samples."""
        return int(self._sample_rate * self.FRAME_DURATION_MS / 1000)

    async def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True

        # Clear any stale data
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        blocksize = self.frame_size
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self.DTYPE,
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            f"Audio capture started: {self._sample_rate}Hz, "
            f"{blocksize} samples/frame ({self.FRAME_DURATION_MS}ms)"
        )

    async def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped")

    async def frames(self) -> AsyncGenerator[np.ndarray, None]:
        """Async generator that yields audio frames.

        Note: For wake word detection, use set_frame_callback() instead
        for lower latency processing.

        Yields:
            Audio frames as int16 numpy arrays.
        """
        while self._running:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield frame
            except asyncio.TimeoutError:
                continue

    async def read_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read a single audio frame.

        Args:
            timeout: Maximum time to wait for a frame.

        Returns:
            Audio frame or None if timeout.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
