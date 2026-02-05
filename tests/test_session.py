"""Tests for session context management."""

import time
from unittest.mock import MagicMock, patch

import pytest

from aggie.context import SessionContext, Turn


class TestTurn:
    """Tests for the Turn dataclass."""

    def test_turn_creation(self):
        """Turn should store role, content, and timestamp."""
        before = time.time()
        turn = Turn(role="user", content="Hello")
        after = time.time()

        assert turn.role == "user"
        assert turn.content == "Hello"
        assert before <= turn.timestamp <= after

    def test_turn_with_explicit_timestamp(self):
        """Turn should accept explicit timestamp."""
        turn = Turn(role="assistant", content="Hi there", timestamp=1000.0)
        assert turn.timestamp == 1000.0


class TestSessionContext:
    """Tests for SessionContext class."""

    def test_initial_state(self):
        """New session should be empty."""
        session = SessionContext()
        assert session.turn_count == 0
        assert session.has_summary is False
        assert session.last_interaction_time is None
        assert session.silence_duration() == 0.0

    def test_add_turn(self):
        """Adding turns should update turn count."""
        session = SessionContext()

        session.add_turn("user", "Hello")
        assert session.turn_count == 1

        session.add_turn("assistant", "Hi there!")
        assert session.turn_count == 2

    def test_add_turn_invalid_role(self):
        """Adding turn with invalid role should raise error."""
        session = SessionContext()

        with pytest.raises(ValueError, match="Invalid role"):
            session.add_turn("system", "Not allowed")

    def test_last_interaction_time(self):
        """Last interaction time should be timestamp of most recent turn."""
        session = SessionContext()

        before = time.time()
        session.add_turn("user", "Hello")
        after = time.time()

        assert before <= session.last_interaction_time <= after

    def test_silence_duration(self):
        """Silence duration should increase over time."""
        session = SessionContext()
        session.add_turn("user", "Hello")

        # Manually set an old timestamp
        session._turns[0].timestamp = time.time() - 60  # 1 minute ago

        silence = session.silence_duration()
        assert 59 <= silence <= 61

    def test_estimate_tokens(self):
        """Token estimation should be based on character count."""
        session = SessionContext()

        # 40 chars = ~10 tokens (4 chars per token)
        session.add_turn("user", "a" * 40)
        assert session.estimate_tokens() == 10

        # Add 80 more chars = ~20 more tokens
        session.add_turn("assistant", "b" * 80)
        assert session.estimate_tokens() == 30

    def test_build_messages_empty(self):
        """Empty session should return empty messages list."""
        session = SessionContext()
        assert session.build_messages() == []

    def test_build_messages_basic(self):
        """Messages should be formatted for Claude API."""
        session = SessionContext()
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi there!")

        messages = session.build_messages()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_build_messages_with_summary(self):
        """When summary exists, it should be prepended."""
        session = SessionContext()
        session._summary = "User asked about weather."
        session.add_turn("user", "What about tomorrow?")

        messages = session.build_messages()

        assert len(messages) == 3
        assert "Previous conversation summary" in messages[0]["content"]
        assert "User asked about weather" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert messages[2] == {"role": "user", "content": "What about tomorrow?"}

    def test_clear(self):
        """Clear should reset all state."""
        session = SessionContext()
        session.add_turn("user", "Hello")
        session._summary = "Some summary"

        session.clear()

        assert session.turn_count == 0
        assert session.has_summary is False

    def test_get_status(self):
        """Status should return current session metrics."""
        session = SessionContext()
        session.add_turn("user", "Hello" * 10)  # 50 chars = 12 tokens

        status = session.get_status()

        assert status["turn_count"] == 1
        assert status["token_estimate"] == 12
        assert status["has_summary"] is False
        assert status["silence_seconds"] >= 0


class TestSessionDecay:
    """Tests for time-based decay logic."""

    def test_no_decay_when_empty(self):
        """Empty session should not trigger decay."""
        session = SessionContext(soft_decay_minutes=1, hard_decay_minutes=2)
        assert session.check_decay() == "none"

    def test_no_decay_within_soft_threshold(self):
        """Recent activity should not trigger decay."""
        session = SessionContext(soft_decay_minutes=1, hard_decay_minutes=2)
        session.add_turn("user", "Hello")

        assert session.check_decay() == "none"

    def test_soft_decay_triggers_summarization(self):
        """Silence beyond soft threshold should trigger summarization."""
        session = SessionContext(soft_decay_minutes=1, hard_decay_minutes=5)

        # Add enough turns to summarize
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi!")
        session.add_turn("user", "How are you?")
        session.add_turn("assistant", "I'm good!")

        # Set timestamp to 2 minutes ago
        for turn in session._turns:
            turn.timestamp = time.time() - 120

        # Mock the summarizer
        with patch.object(session, "_ensure_summarizer") as mock_ensure:
            mock_summarizer = MagicMock()
            mock_summarizer.summarize.return_value = "Greeting exchange."
            mock_ensure.return_value = mock_summarizer

            result = session.check_decay()

        assert result == "soft"
        assert session.has_summary is True

    def test_hard_decay_clears_context(self):
        """Silence beyond hard threshold should clear everything."""
        session = SessionContext(soft_decay_minutes=1, hard_decay_minutes=2)
        session.add_turn("user", "Hello")
        session._summary = "Previous context"

        # Set timestamp to 3 minutes ago
        session._turns[0].timestamp = time.time() - 180

        result = session.check_decay()

        assert result == "hard"
        assert session.turn_count == 0
        assert session.has_summary is False

    def test_soft_decay_only_once(self):
        """Soft decay should not re-trigger if already summarized."""
        session = SessionContext(soft_decay_minutes=1, hard_decay_minutes=5)
        session.add_turn("user", "Hello")
        session._summary = "Already summarized"

        # Set timestamp to 2 minutes ago (past soft threshold)
        session._turns[0].timestamp = time.time() - 120

        # Should not trigger soft decay again
        result = session.check_decay()
        assert result == "none"


class TestSessionSummarization:
    """Tests for summarization logic."""

    def test_summarize_oldest_keeps_recent_turns(self):
        """Summarization should keep the 2 most recent turns."""
        session = SessionContext()

        # Add 4 turns
        session.add_turn("user", "Turn 1")
        session.add_turn("assistant", "Response 1")
        session.add_turn("user", "Turn 2")
        session.add_turn("assistant", "Response 2")

        # Mock the summarizer
        with patch.object(session, "_ensure_summarizer") as mock_ensure:
            mock_summarizer = MagicMock()
            mock_summarizer.summarize.return_value = "Summary of turns 1-2."
            mock_ensure.return_value = mock_summarizer

            session._summarize_oldest()

        # Should keep only last 2 turns
        assert session.turn_count == 2
        assert session._turns[0].content == "Turn 2"
        assert session._turns[1].content == "Response 2"
        assert session._summary == "Summary of turns 1-2."

    def test_no_summarize_if_too_few_turns(self):
        """Should not summarize if 2 or fewer turns."""
        session = SessionContext()
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi!")

        # Should not crash or change anything
        session._summarize_oldest()

        assert session.turn_count == 2
        assert session.has_summary is False

    def test_token_limit_triggers_summarization(self):
        """Exceeding token limit should trigger summarization."""
        session = SessionContext(max_session_tokens=50)

        # Mock the summarizer before adding turns
        with patch.object(session, "_ensure_summarizer") as mock_ensure:
            mock_summarizer = MagicMock()
            mock_summarizer.summarize.return_value = "Summary."
            mock_ensure.return_value = mock_summarizer

            # Add turns that exceed token limit
            # 200 chars each = 50 tokens each, limit is 50
            session.add_turn("user", "x" * 200)
            session.add_turn("assistant", "y" * 200)
            session.add_turn("user", "z" * 200)  # This should trigger summarization

        # Summarization should have been triggered
        assert mock_summarizer.summarize.called
