"""Speech-to-text using faster-whisper."""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def detect_gpu() -> Tuple[str, Optional[str]]:
    """Detect best available device and return device info.

    Uses ctranslate2 (faster-whisper's backend) for detection, with torch fallback
    for detailed GPU info.

    Returns:
        Tuple of (device, gpu_info) where device is 'cuda' or 'cpu',
        and gpu_info is a description string or None.
    """
    # Check if ctranslate2 supports CUDA (faster-whisper's backend)
    try:
        import ctranslate2

        cuda_available = "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except (ImportError, RuntimeError):
        cuda_available = False

    if not cuda_available:
        return "cpu", None

    # Try to get detailed GPU info via torch (optional)
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return "cuda", f"{gpu_name} ({gpu_memory:.1f}GB)"
    except ImportError:
        pass

    # CUDA available but no torch for details
    return "cuda", "CUDA (install torch for GPU details)"


class SpeechToText:
    """Transcribes audio using faster-whisper.

    The model is loaded on-demand to save memory when idle.
    Supports automatic device selection (CPU/CUDA).
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
    ) -> None:
        """Initialize speech-to-text.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
            device: Device to use (auto, cpu, cuda).
            compute_type: Compute type (auto, int8, float16).
            language: Language code for transcription.
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None

    def _ensure_model(self) -> None:
        """Lazy-load the whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        # Determine device
        device = self._device
        gpu_info = None
        if device == "auto":
            device, gpu_info = detect_gpu()

        # Determine compute type
        compute_type = self._compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        # Log device selection with context
        if device == "cuda" and gpu_info:
            logger.info(
                f"Loading Whisper model: {self._model_size} on GPU",
                extra={"ctx": {"device": device, "gpu": gpu_info, "compute_type": compute_type}}
            )
        else:
            logger.info(
                f"Loading Whisper model: {self._model_size} on CPU",
                extra={"ctx": {"device": device, "compute_type": compute_type}}
            )

        self._model = WhisperModel(
            self._model_size,
            device=device,
            compute_type=compute_type,
        )

        if device == "cuda":
            logger.info(f"Whisper ready: GPU acceleration enabled ({gpu_info})")
        else:
            logger.info("Whisper ready: CPU mode (set device: cuda in config to force GPU)")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as int16 or float32 numpy array.
            sample_rate: Sample rate of the audio (should be 16000).

        Returns:
            Transcribed text.
        """
        self._ensure_model()

        # Convert to float32 normalized audio (required by faster-whisper)
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        # Flatten if needed
        if audio_float.ndim > 1:
            audio_float = audio_float.flatten()

        logger.debug(f"Transcribing {len(audio_float) / sample_rate:.1f}s of audio")

        # Transcribe with VAD filter for better results
        segments, info = self._model.transcribe(
            audio_float,
            language=self._language,
            vad_filter=True,
            beam_size=5,
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcript = " ".join(text_parts).strip()
        logger.info(f"Transcription: '{transcript[:100]}{'...' if len(transcript) > 100 else ''}'")

        return transcript

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            self._model = None
            logger.info("Whisper model unloaded")

    @property
    def is_loaded(self) -> bool:
        """True if model is currently loaded."""
        return self._model is not None
