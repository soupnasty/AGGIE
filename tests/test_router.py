"""Tests for query router."""

import pytest
from datetime import datetime
from aggie.router import (
    Router,
    ResponseSource,
    check_system_command,
)


class TestSystemCommands:
    """Tests for system command detection."""

    def test_time_query(self):
        """Test time queries are detected."""
        queries = [
            "What time is it?",
            "What's the time?",
            "Tell me the time",
            "What time is it now?",
            "current time please",
        ]
        for query in queries:
            result = check_system_command(query)
            assert result is not None, f"Failed to match: {query}"
            assert result.command == "time"
            assert ":" in result.text  # Contains time format

    def test_date_query(self):
        """Test date queries are detected."""
        queries = [
            "What's the date?",
            "What date is it?",
            "What day is it?",
            "What day is today?",
            "Today's date please",
        ]
        for query in queries:
            result = check_system_command(query)
            assert result is not None, f"Failed to match: {query}"
            assert result.command == "date"
            # Should contain day name
            assert any(day in result.text for day in [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ])

    def test_non_system_query(self):
        """Test non-system queries return None."""
        queries = [
            "What's the weather?",
            "Tell me a joke",
            "How do I make coffee?",
            "What's the capital of France?",
        ]
        for query in queries:
            result = check_system_command(query)
            assert result is None, f"Incorrectly matched: {query}"


class TestRouter:
    """Tests for Router class."""

    def test_route_system_command(self):
        """Test system commands are routed correctly."""
        router = Router()
        decision = router.route("What time is it?")
        assert decision.source == ResponseSource.SYSTEM
        assert decision.system_response is not None

    def test_route_claude_trigger_use_claude(self):
        """Test 'use claude' trigger."""
        router = Router()
        decision = router.route("Use Claude to explain quantum physics")
        assert decision.source == ResponseSource.CLAUDE
        assert decision.query == "to explain quantum physics"

    def test_route_claude_trigger_ask_claude(self):
        """Test 'ask claude' trigger."""
        router = Router()
        decision = router.route("Ask Claude about the weather")
        assert decision.source == ResponseSource.CLAUDE
        assert decision.query == "about the weather"

    def test_route_claude_trigger_prefix(self):
        """Test 'claude,' prefix trigger."""
        router = Router()
        decision = router.route("Claude, what do you think?")
        assert decision.source == ResponseSource.CLAUDE
        assert decision.query == "what do you think?"

    def test_route_default_to_llama(self):
        """Test default routing to Llama."""
        router = Router()
        queries = [
            "What's the capital of France?",
            "Tell me a joke",
            "Explain photosynthesis",
            "How do I make pasta?",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.source == ResponseSource.LLAMA, f"Failed for: {query}"
            assert decision.query == query

    def test_route_claude_trigger_case_insensitive(self):
        """Test Claude triggers are case insensitive."""
        router = Router()
        queries = [
            "USE CLAUDE for this",
            "Use Claude for this",
            "use claude for this",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.source == ResponseSource.CLAUDE

    def test_route_trigger_must_be_at_start(self):
        """Test Claude triggers only work at start of query."""
        router = Router()
        # These should NOT trigger Claude routing
        queries = [
            "I want to use claude for this",
            "Can you ask claude about this?",
            "Please claude help me",
        ]
        for query in queries:
            decision = router.route(query)
            assert decision.source == ResponseSource.LLAMA, f"Incorrectly matched: {query}"
