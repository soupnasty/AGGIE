"""Session context management for multi-turn conversations."""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .summarizer import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class SessionContext:
    """Manages conversation context with time-based decay.

    Tracks conversation turns with timestamps, handles soft/hard decay
    based on silence duration, and triggers local summarization when
    needed to stay within token limits.
    """

    # Rough estimate: ~4 characters per token for Claude
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        soft_decay_minutes: int = 10,
        hard_decay_minutes: int = 30,
        max_session_tokens: int = 8000,
        summarizer_model: str = "facebook/bart-large-cnn",
    ) -> None:
        """Initialize session context.

        Args:
            soft_decay_minutes: Minutes of silence before summarizing old turns.
            hard_decay_minutes: Minutes of silence before clearing all context.
            max_session_tokens: Token ceiling before forced summarization.
            summarizer_model: HuggingFace model ID for local summarization.
        """
        self._soft_decay_seconds = soft_decay_minutes * 60
        self._hard_decay_seconds = hard_decay_minutes * 60
        self._max_tokens = max_session_tokens
        self._summarizer_model = summarizer_model

        self._turns: list[Turn] = []
        self._summary: Optional[str] = None  # Summary of older turns
        self._summarizer: Optional["Summarizer"] = None

    @property
    def turn_count(self) -> int:
        """Number of turns in current session."""
        return len(self._turns)

    @property
    def has_summary(self) -> bool:
        """Whether older turns have been summarized."""
        return self._summary is not None

    @property
    def last_interaction_time(self) -> Optional[float]:
        """Timestamp of most recent turn, or None if empty."""
        if not self._turns:
            return None
        return self._turns[-1].timestamp

    def silence_duration(self) -> float:
        """Seconds since last interaction."""
        if not self._turns:
            return 0.0
        return time.time() - self._turns[-1].timestamp

    def estimate_tokens(self) -> int:
        """Estimate total token count for current context."""
        total_chars = 0
        if self._summary:
            total_chars += len(self._summary)
        for turn in self._turns:
            total_chars += len(turn.content)
        return total_chars // self.CHARS_PER_TOKEN

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn.

        Args:
            role: "user" or "assistant"
            content: The message content.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}")

        self._turns.append(Turn(role=role, content=content))
        logger.debug(
            f"Added {role} turn ({len(content)} chars), "
            f"session now has {len(self._turns)} turns, "
            f"~{self.estimate_tokens()} tokens"
        )

        # Check if we need to summarize due to token limit
        if self.estimate_tokens() > self._max_tokens:
            logger.info(
                f"Token limit exceeded ({self.estimate_tokens()} > {self._max_tokens}), "
                "summarizing oldest turns"
            )
            self._summarize_oldest()

    def check_decay(self) -> str:
        """Check and apply time-based decay.

        Should be called before each new interaction.

        Returns:
            Decay action taken: "none", "soft", or "hard"
        """
        if not self._turns:
            return "none"

        silence = self.silence_duration()

        if silence >= self._hard_decay_seconds:
            logger.info(
                f"Hard decay: {silence / 60:.1f} min silence >= "
                f"{self._hard_decay_seconds / 60:.0f} min threshold, clearing context"
            )
            self.clear()
            return "hard"

        if silence >= self._soft_decay_seconds and not self._summary:
            logger.info(
                f"Soft decay: {silence / 60:.1f} min silence >= "
                f"{self._soft_decay_seconds / 60:.0f} min threshold, summarizing"
            )
            self._summarize_oldest()
            return "soft"

        return "none"

    def build_messages(self) -> list[dict]:
        """Build messages list for Claude API.

        Returns:
            List of message dicts with 'role' and 'content' keys.
            If there's a summary, it's prepended as a system-style user message.
        """
        messages = []

        # Include summary as context if present
        if self._summary:
            messages.append({
                "role": "user",
                "content": f"[Previous conversation summary: {self._summary}]",
            })
            messages.append({
                "role": "assistant",
                "content": "I understand. I'll keep that context in mind.",
            })

        # Add current turns
        for turn in self._turns:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages

    def clear(self) -> None:
        """Clear all context (hard reset)."""
        self._turns = []
        self._summary = None
        logger.debug("Session context cleared")

    def get_status(self) -> dict:
        """Get current session status for IPC/debugging.

        Returns:
            Dict with turn_count, token_estimate, silence_seconds, has_summary.
        """
        return {
            "turn_count": len(self._turns),
            "token_estimate": self.estimate_tokens(),
            "silence_seconds": self.silence_duration(),
            "has_summary": self._summary is not None,
        }

    def _ensure_summarizer(self) -> "Summarizer":
        """Lazy-load the summarizer."""
        if self._summarizer is None:
            from .summarizer import Summarizer
            logger.info(f"Loading summarizer model: {self._summarizer_model}")
            self._summarizer = Summarizer(model_name=self._summarizer_model)
        return self._summarizer

    def _summarize_oldest(self) -> None:
        """Summarize oldest turns to reduce context size.

        Keeps the most recent 2 turns verbatim, summarizes the rest.
        """
        if len(self._turns) <= 2:
            # Not enough turns to summarize
            return

        # Split: older turns to summarize, recent turns to keep
        turns_to_summarize = self._turns[:-2]
        turns_to_keep = self._turns[-2:]

        # Build text to summarize
        conversation_text = ""
        if self._summary:
            conversation_text += f"Previous context: {self._summary}\n\n"

        for turn in turns_to_summarize:
            speaker = "User" if turn.role == "user" else "Assistant"
            conversation_text += f"{speaker}: {turn.content}\n"

        # Run summarization
        summarizer = self._ensure_summarizer()
        self._summary = summarizer.summarize(conversation_text)

        # Keep only recent turns
        self._turns = turns_to_keep

        logger.info(
            f"Summarized {len(turns_to_summarize)} turns into {len(self._summary)} chars, "
            f"keeping {len(turns_to_keep)} recent turns"
        )
