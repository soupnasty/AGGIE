"""Local summarization for conversation context compression."""

import logging

logger = logging.getLogger(__name__)


class Summarizer:
    """Local text summarizer using HuggingFace transformers.

    Runs on CPU to avoid GPU contention with faster-whisper.
    Lazy-loads the model on first use.
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn") -> None:
        """Initialize summarizer.

        Args:
            model_name: HuggingFace model ID for summarization.
        """
        self._model_name = model_name
        self._pipeline = None

    def _ensure_pipeline(self):
        """Lazy-load the summarization pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading summarization model: {self._model_name}")
            from transformers import pipeline

            # Force CPU to avoid GPU contention with faster-whisper
            self._pipeline = pipeline(
                "summarization",
                model=self._model_name,
                device=-1,  # CPU
            )
            logger.info("Summarization model loaded")
        return self._pipeline

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Summarize the given text.

        Args:
            text: Text to summarize (conversation transcript).
            max_length: Maximum length of summary in tokens.
            min_length: Minimum length of summary in tokens.

        Returns:
            Summarized text.
        """
        if not text.strip():
            return ""

        pipe = self._ensure_pipeline()

        # BART has a max input length of 1024 tokens
        # Truncate if needed (rough estimate: 4 chars per token)
        max_input_chars = 1024 * 4
        if len(text) > max_input_chars:
            logger.warning(
                f"Input text ({len(text)} chars) exceeds model limit, truncating"
            )
            text = text[:max_input_chars]

        try:
            result = pipe(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            summary = result[0]["summary_text"]
            logger.debug(f"Summarized {len(text)} chars -> {len(summary)} chars")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: return truncated original
            return text[:500] + "..."
