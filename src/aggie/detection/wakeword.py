"""Wake word detection using openWakeWord."""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Detects wake word using openWakeWord with Silero VAD.

    The detector processes audio frames and returns detection results.
    It uses the integrated Silero VAD to reduce false positives from
    non-speech sounds.
    """

    def __init__(
        self,
        model_path: str = "hey_jarvis",
        threshold: float = 0.5,
        vad_threshold: float = 0.5,
        enable_speex: bool = False,  # Disabled by default - requires speexdsp_ns
    ) -> None:
        """Initialize wake word detector.

        Args:
            model_path: Pre-trained model name or path to custom model.
            threshold: Detection threshold (0.0-1.0).
            vad_threshold: VAD threshold for noise rejection.
            enable_speex: Enable Speex noise suppression (Linux only).
        """
        import openwakeword
        from openwakeword.model import Model

        # Ensure models are downloaded
        logger.info("Checking/downloading wake word models...")
        openwakeword.utils.download_models()

        self._threshold = threshold
        self._model_name = self._normalize_model_name(model_path)

        logger.info(f"Loading wake word model: {model_path}")
        self._model = Model(
            wakeword_models=[model_path],
            vad_threshold=vad_threshold,
            enable_speex_noise_suppression=enable_speex,
        )
        logger.info(f"Wake word detector ready (threshold={threshold})")

    def _normalize_model_name(self, model_path: str) -> str:
        """Normalize model path to model name for prediction lookup."""
        # openWakeWord uses the model filename (without extension) as the key
        name = model_path.replace("/", "_").replace("\\", "_")
        if name.endswith(".onnx"):
            name = name[:-5]
        if name.endswith(".tflite"):
            name = name[:-7]
        return name

    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """Process an audio frame and check for wake word.

        Args:
            audio_frame: 16-bit 16kHz mono PCM audio (80ms frame recommended).

        Returns:
            Tuple of (detected: bool, confidence: float).
        """
        # Ensure correct format
        if audio_frame.dtype != np.int16:
            audio_frame = audio_frame.astype(np.int16)

        # Flatten if needed
        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()

        # Get prediction
        prediction = self._model.predict(audio_frame)

        # Only log high confidence predictions to reduce noise
        max_conf = max(prediction.values()) if prediction else 0
        if max_conf > 0.1:
            logger.debug(f"[WAKEWORD] prediction: {prediction}")

        # Extract score for our wake word
        # The model returns a dict with model names as keys
        confidence = 0.0
        for key, value in prediction.items():
            if self._model_name in key or key in self._model_name:
                # Convert to float - value may be numpy.float32 which isn't isinstance of float
                try:
                    confidence = float(value)
                except (TypeError, ValueError):
                    confidence = 0.0
                break
            else:
                # Log if key doesn't match to help debug
                logger.debug(f"[WAKEWORD] Key '{key}' doesn't match '{self._model_name}'")

        detected = confidence >= self._threshold

        if detected:
            logger.info(f"Wake word detected! Confidence: {confidence:.3f}")
            self._model.reset()

        return detected, confidence

    def reset(self) -> None:
        """Reset the model state.

        Should be called after a detection to prepare for the next one.
        """
        self._model.reset()

    @property
    def threshold(self) -> float:
        """Current detection threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set detection threshold."""
        self._threshold = max(0.0, min(1.0, value))
