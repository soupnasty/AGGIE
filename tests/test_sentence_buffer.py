"""Tests for sentence buffer."""

import pytest
from aggie.llm.sentence_buffer import SentenceBuffer, sentences_from_stream


class TestSentenceBuffer:
    """Tests for SentenceBuffer class."""

    def test_simple_sentence(self):
        """Test extracting a simple sentence."""
        buffer = SentenceBuffer(min_words=1)
        sentences = buffer.add("Hello world. ")
        assert sentences == ["Hello world."]

    def test_multiple_sentences(self):
        """Test extracting multiple sentences at once."""
        buffer = SentenceBuffer(min_words=1)
        sentences = buffer.add("First sentence. Second sentence. ")
        assert sentences == ["First sentence.", "Second sentence."]

    def test_partial_sentence(self):
        """Test that partial sentences are buffered."""
        buffer = SentenceBuffer(min_words=1)
        sentences = buffer.add("This is a partial")
        assert sentences == []

        # Complete the sentence
        sentences = buffer.add(" sentence. ")
        assert sentences == ["This is a partial sentence."]

    def test_streaming_chunks(self):
        """Test accumulating text from small chunks."""
        buffer = SentenceBuffer(min_words=1)

        # Simulate streaming tokens
        assert buffer.add("Hello") == []
        assert buffer.add(" ") == []
        assert buffer.add("world") == []
        assert buffer.add(".") == []  # Period alone doesn't trigger
        assert buffer.add(" ") == ["Hello world."]  # Space after period triggers

    def test_question_mark(self):
        """Test question mark as sentence boundary."""
        buffer = SentenceBuffer(min_words=1)
        sentences = buffer.add("How are you? ")
        assert sentences == ["How are you?"]

    def test_exclamation_mark(self):
        """Test exclamation mark as sentence boundary."""
        buffer = SentenceBuffer(min_words=1)
        sentences = buffer.add("Great news! ")
        assert sentences == ["Great news!"]

    def test_min_words_filter(self):
        """Test minimum word filtering."""
        buffer = SentenceBuffer(min_words=3)

        # Two words - should not emit
        sentences = buffer.add("Hi. ")
        assert sentences == []

        # With more text, should combine
        sentences = buffer.add("How are you today? ")
        assert sentences == ["Hi. How are you today?"]

    def test_flush(self):
        """Test flushing remaining text."""
        buffer = SentenceBuffer(min_words=1)
        buffer.add("Incomplete sentence")
        remaining = buffer.flush()
        assert remaining == "Incomplete sentence"

    def test_flush_empty(self):
        """Test flushing empty buffer."""
        buffer = SentenceBuffer()
        assert buffer.flush() == ""

    def test_mixed_punctuation(self):
        """Test mixed sentence endings."""
        buffer = SentenceBuffer(min_words=1)
        sentences = buffer.add("Statement. Question? Exclamation! ")
        assert sentences == ["Statement.", "Question?", "Exclamation!"]


@pytest.mark.asyncio
async def test_sentences_from_stream():
    """Test the async sentence stream converter."""

    async def token_stream():
        tokens = ["Hello ", "world. ", "How ", "are ", "you? ", "Great!"]
        for token in tokens:
            yield token

    sentences = []
    async for sentence in sentences_from_stream(token_stream(), min_words=1):
        sentences.append(sentence)

    assert sentences == ["Hello world.", "How are you?", "Great!"]


@pytest.mark.asyncio
async def test_sentences_from_stream_flush():
    """Test that incomplete sentences are flushed at end."""

    async def token_stream():
        yield "This has no ending punctuation"

    sentences = []
    async for sentence in sentences_from_stream(token_stream(), min_words=1):
        sentences.append(sentence)

    assert sentences == ["This has no ending punctuation"]
