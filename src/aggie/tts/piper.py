"""Text-to-speech using Piper."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Synthesizes speech using Piper TTS.

    Uses the piper command-line tool for synthesis, which provides
    more reliable operation than the Python bindings.
    """

    def __init__(
        self,
        voice_model: str = "en_US-lessac-medium",
        speaking_rate: float = 1.0,
        use_cuda: bool = False,
    ) -> None:
        """Initialize text-to-speech.

        Args:
            voice_model: Piper voice model name or path.
            speaking_rate: Speaking rate multiplier (0.5=slow, 2.0=fast).
            use_cuda: Use CUDA for synthesis (if available).
        """
        self._voice_model = voice_model
        self._speaking_rate = speaking_rate
        self._use_cuda = use_cuda
        self._piper_path: Optional[str] = None
        self._model_path: Optional[Path] = None

    def _find_piper(self) -> str:
        """Find the piper executable."""
        if self._piper_path:
            return self._piper_path

        import shutil

        # Try to find piper in PATH
        piper = shutil.which("piper")
        if piper:
            self._piper_path = piper
            return piper

        # Try common locations
        common_paths = [
            Path.home() / ".local" / "bin" / "piper",
            Path("/usr/local/bin/piper"),
            Path("/usr/bin/piper"),
        ]
        for path in common_paths:
            if path.exists():
                self._piper_path = str(path)
                return self._piper_path

        raise FileNotFoundError(
            "Piper executable not found. Install with: pip install piper-tts"
        )

    def _get_model_path(self) -> Path:
        """Get or download the voice model path."""
        if self._model_path and self._model_path.exists():
            return self._model_path

        # Check if it's already a path
        model_path = Path(self._voice_model)
        if model_path.exists():
            self._model_path = model_path
            return model_path

        # Check in piper data directory
        data_dir = Path.home() / ".local" / "share" / "piper-voices"

        # Check flat layout: data_dir/<voice>.onnx
        flat_onnx = data_dir / f"{self._voice_model}.onnx"
        if flat_onnx.exists():
            self._model_path = flat_onnx
            return flat_onnx

        # Check nested layout: data_dir/<voice>/<voice>.onnx
        nested_onnx = data_dir / self._voice_model / f"{self._voice_model}.onnx"
        if nested_onnx.exists():
            self._model_path = nested_onnx
            return nested_onnx

        # Try to download the model
        logger.info(f"Downloading Piper voice: {self._voice_model}")
        try:
            from piper.download import ensure_voice_exists, find_voice, get_voices

            # Get available voices
            voices = get_voices(data_dir, update_voices=True)
            voice_info = find_voice(self._voice_model, voices)

            if voice_info:
                # Download if needed
                ensure_voice_exists(
                    self._voice_model,
                    data_dir,
                    data_dir,
                    voices,
                )
                # Find the downloaded model
                for ext in [".onnx"]:
                    for f in data_dir.rglob(f"*{ext}"):
                        if self._voice_model in str(f):
                            self._model_path = f
                            return f

        except Exception as e:
            logger.warning(f"Could not download voice model: {e}")

        raise FileNotFoundError(
            f"Voice model not found: {self._voice_model}. "
            "Download with: piper --download-voice"
        )

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to speak.

        Returns:
            Tuple of (audio_data as int16 numpy array, sample_rate).
        """
        if not text.strip():
            return np.array([], dtype=np.int16), 22050

        logger.debug(f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Use piper-tts Python package
        try:
            return self._synthesize_with_piper_tts(text)
        except Exception as e:
            logger.warning(f"piper-tts failed: {e}, trying CLI")
            return self._synthesize_with_cli(text)

    def _synthesize_with_piper_tts(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using piper-tts Python package."""
        from piper import PiperVoice

        model_path = self._get_model_path()
        voice = PiperVoice.load(str(model_path), use_cuda=self._use_cuda)

        # Synthesize — returns AudioChunk objects with audio_int16_bytes
        audio_bytes = b""
        for chunk in voice.synthesize(text):
            audio_bytes += chunk.audio_int16_bytes

        # Convert to numpy array
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        # Get sample rate from voice config
        sample_rate = voice.config.sample_rate

        logger.debug(f"Synthesized {len(audio)} samples at {sample_rate}Hz")
        return audio, sample_rate

    def _synthesize_with_cli(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using piper CLI as fallback."""
        piper = self._find_piper()
        model_path = self._get_model_path()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Build command
            cmd = [
                piper,
                "--model",
                str(model_path),
                "--output_file",
                output_path,
            ]

            if self._speaking_rate != 1.0:
                # Piper uses length_scale (inverse of rate)
                cmd.extend(["--length-scale", str(1.0 / self._speaking_rate)])

            # Run piper
            proc = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )

            if proc.returncode != 0:
                raise RuntimeError(f"Piper failed: {proc.stderr.decode()}")

            # Read the output WAV file
            audio, sample_rate = sf.read(output_path, dtype="int16")

            logger.debug(f"Synthesized {len(audio)} samples at {sample_rate}Hz")
            return audio.astype(np.int16), sample_rate

        finally:
            # Clean up temp file
            Path(output_path).unlink(missing_ok=True)

    async def synthesize_stream(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> AsyncIterator[Tuple[np.ndarray, int]]:
        """Synthesize sentences from a stream, yielding audio chunks.

        Processes sentences as they arrive, yielding audio for each
        sentence to enable streaming playback.

        Args:
            sentence_stream: Async iterator yielding sentences.

        Yields:
            Tuples of (audio_data as int16 numpy array, sample_rate).
        """
        import asyncio

        sentence_count = 0
        async for sentence in sentence_stream:
            if not sentence.strip():
                continue

            sentence_count += 1
            logger.debug(f"Synthesizing sentence {sentence_count}: '{sentence[:40]}...'")

            # Run TTS in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            try:
                audio_data, sample_rate = await loop.run_in_executor(
                    None, self.synthesize, sentence
                )
                if len(audio_data) > 0:
                    yield audio_data, sample_rate
            except Exception as e:
                logger.error(f"Failed to synthesize sentence: {e}")
                # Continue with next sentence rather than failing entirely

        logger.info(f"Streaming TTS complete: {sentence_count} sentences")
