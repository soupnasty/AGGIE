"""Audio playback module using sounddevice."""

import asyncio
import logging
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayback:
    """Plays audio through the default output device.

    Supports cancellation mid-playback via asyncio event.
    """

    def __init__(self) -> None:
        """Initialize audio playback."""
        self._stream: Optional[sd.OutputStream] = None
        self._cancel_event = asyncio.Event()
        self._playback_done = asyncio.Event()
        self._playing = False

    @property
    def is_playing(self) -> bool:
        """True if currently playing audio."""
        return self._playing

    async def play(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Play audio data asynchronously.

        Args:
            audio_data: Audio samples to play (int16 or float32).
            sample_rate: Sample rate of the audio.

        Returns:
            True if playback completed, False if cancelled.
        """
        if len(audio_data) == 0:
            return True

        self._cancel_event.clear()
        self._playback_done.clear()
        self._playing = True

        loop = asyncio.get_running_loop()
        position = 0
        blocksize = 1024
        cancelled = False

        # Ensure audio is the right shape
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(-1, 1)

        def callback(
            outdata: np.ndarray,
            frames: int,
            time_info: dict,
            status: sd.CallbackFlags,
        ) -> None:
            nonlocal position, cancelled

            if self._cancel_event.is_set():
                outdata.fill(0)
                cancelled = True
                raise sd.CallbackStop()

            end = position + frames
            if end >= len(audio_data):
                # End of audio - pad with silence and stop
                remaining = len(audio_data) - position
                if remaining > 0:
                    outdata[:remaining] = audio_data[position:]
                outdata[remaining:] = 0
                raise sd.CallbackStop()
            else:
                outdata[:] = audio_data[position:end]
                position = end

        def finished_callback() -> None:
            loop.call_soon_threadsafe(self._playback_done.set)

        try:
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=audio_data.shape[1] if audio_data.ndim > 1 else 1,
                dtype=audio_data.dtype,
                blocksize=blocksize,
                callback=callback,
                finished_callback=finished_callback,
            )

            logger.debug(f"Starting playback: {len(audio_data)} samples at {sample_rate}Hz")
            self._stream.start()
            await self._playback_done.wait()

        except Exception as e:
            logger.error(f"Playback error: {e}")
            cancelled = True

        finally:
            self._playing = False
            if self._stream:
                self._stream.close()
                self._stream = None

        if cancelled:
            logger.info("Playback cancelled")
            return False

        logger.debug("Playback completed")
        return True

    def cancel(self) -> None:
        """Cancel current playback.

        Playback will stop gracefully at the next audio callback.
        """
        if self._playing:
            logger.info("Cancelling playback")
            self._cancel_event.set()
