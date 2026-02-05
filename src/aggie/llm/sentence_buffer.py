"""Sentence buffer for streaming LLM responses."""

import logging
import re
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Sentence ending patterns - period, question mark, exclamation mark
# followed by space (not end of string, to avoid premature emission during streaming)
SENTENCE_END_PATTERN = re.compile(r'[.!?]\s')

# Minimum words before we'll emit a sentence (avoid tiny fragments)
MIN_WORDS_FOR_SENTENCE = 3


class SentenceBuffer:
    """Accumulates streaming text and yields complete sentences.

    Buffers incoming text chunks until sentence boundaries are detected,
    then yields complete sentences for TTS processing.
    """

    def __init__(self, min_words: int = MIN_WORDS_FOR_SENTENCE) -> None:
        """Initialize the sentence buffer.

        Args:
            min_words: Minimum words before emitting a sentence.
        """
        self._buffer = ""
        self._min_words = min_words

    def add(self, text: str) -> list[str]:
        """Add text to buffer and return any complete sentences.

        Args:
            text: Text chunk to add.

        Returns:
            List of complete sentences (may be empty).
        """
        self._buffer += text
        return self._extract_sentences()

    def flush(self) -> str:
        """Flush any remaining text from the buffer.

        Returns:
            Remaining text, or empty string if buffer is empty.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining

    def _extract_sentences(self) -> list[str]:
        """Extract complete sentences from the buffer.

        Returns:
            List of complete sentences.
        """
        sentences = []

        while True:
            # Find all sentence boundaries and pick the best one
            best_end_pos = None
            search_pos = 0

            while True:
                match = SENTENCE_END_PATTERN.search(self._buffer, search_pos)
                if not match:
                    break

                end_pos = match.end()
                sentence = self._buffer[:end_pos].strip()
                word_count = len(sentence.split())

                if word_count >= self._min_words:
                    # This boundary gives us enough words - emit it
                    best_end_pos = end_pos
                    break
                else:
                    # Not enough words - look for next boundary
                    search_pos = end_pos

            if best_end_pos is None:
                # No suitable boundary found - wait for more text
                break

            # Emit the sentence
            sentence = self._buffer[:best_end_pos].strip()
            sentences.append(sentence)
            self._buffer = self._buffer[best_end_pos:].lstrip()

        return sentences


async def sentences_from_stream(
    token_stream: AsyncIterator[str],
    min_words: int = MIN_WORDS_FOR_SENTENCE,
) -> AsyncIterator[str]:
    """Convert a token stream into a sentence stream.

    Args:
        token_stream: Async iterator yielding text chunks.
        min_words: Minimum words per sentence.

    Yields:
        Complete sentences as they are detected.
    """
    buffer = SentenceBuffer(min_words=min_words)

    async for chunk in token_stream:
        sentences = buffer.add(chunk)
        for sentence in sentences:
            logger.debug(f"Sentence complete: '{sentence[:50]}...'")
            yield sentence

    # Flush any remaining text
    remaining = buffer.flush()
    if remaining:
        logger.debug(f"Flushing remaining: '{remaining[:50]}...'")
        yield remaining
