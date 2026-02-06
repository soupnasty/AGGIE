"""Tests for three-tier context management system."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aggie.context import (
    AgentDef,
    Constraint,
    ContextCompressor,
    Decision,
    ProjectState,
    PromptComposer,
    ToolSchema,
    Turn,
)
from aggie.context.composer import BASE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# TestProjectState
# ---------------------------------------------------------------------------

class TestProjectState:
    def test_initial_state(self):
        state = ProjectState()
        assert state.turn_count == 0
        assert state.history_summary is None
        assert state.recent_messages == []
        assert state.decisions == []
        assert state.agents == []
        assert state.mcp_tools == []
        assert state.constraints == []
        assert state.silence_duration() == 0.0

    def test_add_turn(self):
        state = ProjectState()
        state.add_turn("user", "Hello")
        assert state.turn_count == 1
        assert len(state.recent_messages) == 1
        assert state.recent_messages[0].role == "user"
        assert state.recent_messages[0].content == "Hello"

        state.add_turn("assistant", "Hi there!")
        assert state.turn_count == 2
        assert len(state.recent_messages) == 2

    def test_add_turn_invalid_role(self):
        state = ProjectState()
        with pytest.raises(ValueError, match="Invalid role"):
            state.add_turn("system", "Not allowed")

    def test_silence_duration(self):
        state = ProjectState()
        state.add_turn("user", "Hello")
        state.recent_messages[0].timestamp = time.time() - 60
        silence = state.silence_duration()
        assert 59 <= silence <= 61

    def test_estimate_tokens(self):
        state = ProjectState()
        # 40 chars = 10 tokens
        state.add_turn("user", "a" * 40)
        assert state.estimate_tokens() == 10

        state.add_turn("assistant", "b" * 80)
        assert state.estimate_tokens() == 30

    def test_to_sacred_block_empty(self):
        state = ProjectState()
        assert state.to_sacred_block() == ""

    def test_to_sacred_block_populated(self):
        state = ProjectState()
        state.decisions.append(Decision(summary="Use Haiku", rationale="Cheaper"))
        state.agents.append(AgentDef(name="CodeBot", role="writes code"))
        state.mcp_tools.append(ToolSchema(name="search", description="web search"))
        state.constraints.append(Constraint(description="No PII"))

        block = state.to_sacred_block()
        assert "<sacred_context>" in block
        assert "<decisions>" in block
        assert "Use Haiku" in block
        assert "Cheaper" in block
        assert "<agents>" in block
        assert "CodeBot" in block
        assert "<mcp_tools>" in block
        assert "search" in block
        assert "<constraints>" in block
        assert "No PII" in block

    def test_clear(self):
        state = ProjectState()
        state.add_turn("user", "Hello")
        state.decisions.append(Decision(summary="test"))
        state.history_summary = "some summary"

        state.clear()

        assert state.turn_count == 0
        assert state.recent_messages == []
        assert state.decisions == []
        assert state.history_summary is None

    def test_clear_working(self):
        state = ProjectState()
        state.add_turn("user", "Hello")
        state.decisions.append(Decision(summary="keep me"))
        state.history_summary = "summary"

        state.clear_working()

        assert state.turn_count == 0
        assert state.recent_messages == []
        assert state.history_summary is None
        # Sacred preserved
        assert len(state.decisions) == 1
        assert state.decisions[0].summary == "keep me"

    def test_get_status(self):
        state = ProjectState()
        state.add_turn("user", "Hello" * 10)  # 50 chars = 12 tokens
        state.decisions.append(Decision(summary="test"))

        status = state.get_status()
        assert status["turn_count"] == 1
        assert status["token_estimate"] >= 12
        assert status["has_history_summary"] is False
        assert status["sacred_decisions"] == 1
        assert status["sacred_agents"] == 0
        assert status["silence_seconds"] >= 0


# ---------------------------------------------------------------------------
# TestPromptComposer
# ---------------------------------------------------------------------------

class TestPromptComposer:
    def test_compose_empty_state(self):
        state = ProjectState()
        composer = PromptComposer(state=state)
        system, messages = composer.compose()

        assert system == BASE_SYSTEM_PROMPT
        assert messages == []

    def test_compose_with_messages_only(self):
        state = ProjectState()
        state.add_turn("user", "Hello")
        state.add_turn("assistant", "Hi!")
        composer = PromptComposer(state=state)
        system, messages = composer.compose()

        assert system == BASE_SYSTEM_PROMPT
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}

    def test_compose_with_sacred_content(self):
        state = ProjectState()
        state.decisions.append(Decision(summary="Use Python"))
        state.add_turn("user", "Hello")
        composer = PromptComposer(state=state)
        system, messages = composer.compose()

        assert "<sacred_context>" in system
        assert "Use Python" in system
        assert BASE_SYSTEM_PROMPT in system
        assert len(messages) == 1

    def test_compose_with_history_summary(self):
        state = ProjectState()
        state.history_summary = "User discussed weather."
        state.add_turn("user", "What about tomorrow?")
        composer = PromptComposer(state=state)
        system, messages = composer.compose()

        assert len(messages) == 3
        assert "Previous conversation summary" in messages[0]["content"]
        assert "User discussed weather" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert messages[2] == {"role": "user", "content": "What about tomorrow?"}

    def test_compose_full(self):
        state = ProjectState()
        state.decisions.append(Decision(summary="Use Haiku"))
        state.history_summary = "Previous context established."
        state.add_turn("user", "Continue please")
        state.add_turn("assistant", "Sure thing")
        composer = PromptComposer(state=state)
        system, messages = composer.compose()

        # System has sacred
        assert "<sacred_context>" in system
        # Messages: summary pair + 2 recent
        assert len(messages) == 4

    def test_custom_base_prompt(self):
        state = ProjectState()
        composer = PromptComposer(state=state, base_prompt="Custom prompt")
        system, _ = composer.compose()
        assert system == "Custom prompt"


# ---------------------------------------------------------------------------
# TestContextCompressor
# ---------------------------------------------------------------------------

class TestContextCompressor:
    def test_should_compress_at_interval(self):
        state = ProjectState()
        compressor = ContextCompressor(
            state=state, api_key="test", compress_every=7, recent_window=6
        )
        # Add 7 turns (to hit the interval) with more than recent_window messages
        for i in range(7):
            state.add_turn("user" if i % 2 == 0 else "assistant", f"Message {i}")

        assert compressor.should_compress() is True

    def test_should_compress_not_enough_messages(self):
        state = ProjectState()
        compressor = ContextCompressor(
            state=state, api_key="test", compress_every=7, recent_window=6
        )
        # Manually set turn count to 7 but only add a few messages
        for i in range(4):
            state.add_turn("user" if i % 2 == 0 else "assistant", f"Message {i}")
        # Force turn count to multiple of 7
        state._turn_count = 7

        # Not enough messages (4 <= 6)
        assert compressor.should_compress() is False

    def test_promote_sacred_decisions(self):
        state = ProjectState()
        compressor = ContextCompressor(state=state, api_key="test")

        json_text = json.dumps({
            "decisions": [
                {"summary": "Use Python", "rationale": "Team knows it"},
                {"summary": "Use Haiku", "rationale": "Cost"},
            ]
        })
        compressor._promote_sacred(json_text)

        assert len(state.decisions) == 2
        assert state.decisions[0].summary == "Use Python"
        assert state.decisions[1].summary == "Use Haiku"

    def test_promote_sacred_agents(self):
        state = ProjectState()
        compressor = ContextCompressor(state=state, api_key="test")

        json_text = json.dumps({
            "agents": [{"name": "CodeBot", "role": "writes code"}]
        })
        compressor._promote_sacred(json_text)

        assert len(state.agents) == 1
        assert state.agents[0].name == "CodeBot"

    def test_promote_sacred_dedup(self):
        state = ProjectState()
        state.decisions.append(Decision(summary="Use Python"))
        compressor = ContextCompressor(state=state, api_key="test")

        json_text = json.dumps({
            "decisions": [
                {"summary": "Use Python", "rationale": "duplicate"},
                {"summary": "Use Haiku", "rationale": "new"},
            ]
        })
        compressor._promote_sacred(json_text)

        assert len(state.decisions) == 2  # not 3
        assert state.decisions[0].summary == "Use Python"
        assert state.decisions[1].summary == "Use Haiku"

    def test_promote_sacred_malformed_json(self):
        state = ProjectState()
        compressor = ContextCompressor(state=state, api_key="test")
        # Should not raise, just warn
        compressor._promote_sacred("not valid json")
        assert len(state.decisions) == 0

    def test_promote_sacred_markdown_fences(self):
        state = ProjectState()
        compressor = ContextCompressor(state=state, api_key="test")

        json_text = '```json\n{"decisions": [{"summary": "Fenced"}]}\n```'
        compressor._promote_sacred(json_text)

        assert len(state.decisions) == 1
        assert state.decisions[0].summary == "Fenced"

    @pytest.mark.asyncio
    async def test_run_compression_integration(self):
        """Integration test with mocked AsyncAnthropic."""
        state = ProjectState()
        # Add enough messages to compress
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            state.add_turn(role, f"Message number {i}")

        compressor = ContextCompressor(
            state=state, api_key="test", compress_every=7, recent_window=6
        )

        # Mock the Haiku client
        mock_client = AsyncMock()
        extraction_response = MagicMock()
        extraction_response.content = [
            MagicMock(text=json.dumps({
                "decisions": [{"summary": "Test decision", "rationale": "testing"}]
            }))
        ]
        compression_response = MagicMock()
        compression_response.content = [
            MagicMock(text="Summary of older messages about testing.")
        ]
        mock_client.messages.create = AsyncMock(
            side_effect=[extraction_response, compression_response]
        )
        compressor._client = mock_client

        await compressor._run_compression()

        # Sacred content extracted
        assert len(state.decisions) == 1
        assert state.decisions[0].summary == "Test decision"

        # History summary set
        assert state.history_summary is not None
        assert "Summary" in state.history_summary

        # Recent messages trimmed to recent_window
        assert len(state.recent_messages) == 6

    @pytest.mark.asyncio
    async def test_fire_and_forget_task(self):
        """Verify maybe_trigger_compression returns a task when conditions met."""
        state = ProjectState()
        for i in range(7):
            state.add_turn("user" if i % 2 == 0 else "assistant", f"Msg {i}")

        compressor = ContextCompressor(
            state=state, api_key="test", compress_every=7, recent_window=6
        )

        # Mock _run_compression to avoid real API calls
        compressor._run_compression = AsyncMock()

        task = compressor.maybe_trigger_compression()
        assert task is not None
        await task
        compressor._run_compression.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_task_when_conditions_not_met(self):
        state = ProjectState()
        state.add_turn("user", "Hello")  # Only 1 turn
        compressor = ContextCompressor(
            state=state, api_key="test", compress_every=7, recent_window=6
        )
        task = compressor.maybe_trigger_compression()
        assert task is None


# ---------------------------------------------------------------------------
# TestClaudeClient
# ---------------------------------------------------------------------------

class TestClaudeClient:
    @pytest.mark.asyncio
    async def test_default_system_prompt(self):
        """get_response uses SYSTEM_PROMPT by default."""
        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="Hello!")]
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            MockAnthropic.return_value = mock_client

            from aggie.llm.claude import ClaudeClient

            client = ClaudeClient(api_key="test-key")
            await client.get_response("Hi")

            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["system"] == ClaudeClient.SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        """get_response uses custom system_prompt when provided."""
        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = AsyncMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text="Hello!")]
            mock_client.messages.create = AsyncMock(return_value=mock_message)
            MockAnthropic.return_value = mock_client

            from aggie.llm.claude import ClaudeClient

            client = ClaudeClient(api_key="test-key")
            await client.get_response("Hi", system_prompt="Custom prompt")

            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["system"] == "Custom prompt"


# ---------------------------------------------------------------------------
# TestClearPhrases
# ---------------------------------------------------------------------------

class TestClearPhrases:
    """Test voice-triggered context clear phrase detection."""

    def _make_daemon_phrases(self):
        """Return the set of clear phrases matching AggieDaemon."""
        return {
            "fresh start",
            "new conversation",
            "forget everything",
            "start over",
            "self destruct",
        }

    def _is_clear_phrase(self, transcript, phrases):
        normalized = transcript.strip().lower().rstrip(".!?")
        return normalized in phrases

    def test_exact_match(self):
        phrases = self._make_daemon_phrases()
        for phrase in phrases:
            assert self._is_clear_phrase(phrase, phrases), f"Should match: {phrase}"

    def test_trailing_punctuation(self):
        phrases = self._make_daemon_phrases()
        assert self._is_clear_phrase("fresh start.", phrases)
        assert self._is_clear_phrase("self destruct!", phrases)
        assert self._is_clear_phrase("start over?", phrases)

    def test_mixed_case(self):
        phrases = self._make_daemon_phrases()
        assert self._is_clear_phrase("Fresh Start", phrases)
        assert self._is_clear_phrase("SELF DESTRUCT", phrases)
        assert self._is_clear_phrase("New Conversation.", phrases)

    def test_whitespace(self):
        phrases = self._make_daemon_phrases()
        assert self._is_clear_phrase("  fresh start  ", phrases)

    def test_non_matching(self):
        phrases = self._make_daemon_phrases()
        assert not self._is_clear_phrase("what's the weather", phrases)
        assert not self._is_clear_phrase("start over please", phrases)
        assert not self._is_clear_phrase("", phrases)
