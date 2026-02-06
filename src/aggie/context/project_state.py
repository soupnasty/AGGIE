"""Three-tier project state for context management.

Tiers:
  Sacred   — decisions, agents, tools, constraints (never summarized)
  Working  — history summary + recent messages (compressed periodically)
  Disposable — older messages dropped during compression
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Decision:
    """A project decision extracted from conversation."""

    summary: str
    rationale: str = ""


@dataclass
class AgentDef:
    """An agent definition extracted from conversation."""

    name: str
    role: str = ""


@dataclass
class ToolSchema:
    """An MCP tool schema extracted from conversation."""

    name: str
    description: str = ""


@dataclass
class Constraint:
    """A project constraint extracted from conversation."""

    description: str


class ProjectState:
    """Three-tier project state: sacred, working, disposable.

    Sacred content is never summarized or dropped.
    Working content (history summary + recent messages) gets compressed.
    """

    def __init__(self) -> None:
        # Sacred tier
        self.decisions: list[Decision] = []
        self.agents: list[AgentDef] = []
        self.mcp_tools: list[ToolSchema] = []
        self.constraints: list[Constraint] = []

        # Working tier
        self.history_summary: Optional[str] = None
        self.recent_messages: list[Turn] = []

        # Metadata
        self.meta: dict = {"project": "default", "phase": ""}

        # Internal counter
        self._turn_count: int = 0

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def add_turn(self, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}")
        self.recent_messages.append(Turn(role=role, content=content))
        self._turn_count += 1
        logger.debug(
            f"Added {role} turn ({len(content)} chars), "
            f"turn_count={self._turn_count}, "
            f"~{self.estimate_tokens()} tokens"
        )

    def to_sacred_block(self) -> str:
        """Serialize sacred content as XML-tagged block for prompt injection."""
        parts: list[str] = []

        if self.decisions:
            items = "\n".join(
                f"- {d.summary}" + (f" ({d.rationale})" if d.rationale else "")
                for d in self.decisions
            )
            parts.append(f"<decisions>\n{items}\n</decisions>")

        if self.agents:
            items = "\n".join(
                f"- {a.name}" + (f": {a.role}" if a.role else "")
                for a in self.agents
            )
            parts.append(f"<agents>\n{items}\n</agents>")

        if self.mcp_tools:
            items = "\n".join(
                f"- {t.name}" + (f": {t.description}" if t.description else "")
                for t in self.mcp_tools
            )
            parts.append(f"<mcp_tools>\n{items}\n</mcp_tools>")

        if self.constraints:
            items = "\n".join(f"- {c.description}" for c in self.constraints)
            parts.append(f"<constraints>\n{items}\n</constraints>")

        if not parts:
            return ""
        return "<sacred_context>\n" + "\n".join(parts) + "\n</sacred_context>"

    def clear(self) -> None:
        """Full hard reset — all tiers."""
        self.decisions.clear()
        self.agents.clear()
        self.mcp_tools.clear()
        self.constraints.clear()
        self.history_summary = None
        self.recent_messages.clear()
        self._turn_count = 0
        logger.debug("ProjectState cleared (all tiers)")

    def clear_working(self) -> None:
        """Preserve sacred, clear working tier."""
        self.history_summary = None
        self.recent_messages.clear()
        self._turn_count = 0
        logger.debug("ProjectState working tier cleared (sacred preserved)")

    def get_status(self) -> dict:
        """Status dict for IPC / debugging."""
        return {
            "turn_count": self._turn_count,
            "token_estimate": self.estimate_tokens(),
            "silence_seconds": self.silence_duration(),
            "has_history_summary": self.history_summary is not None,
            "sacred_decisions": len(self.decisions),
            "sacred_agents": len(self.agents),
            "sacred_tools": len(self.mcp_tools),
            "sacred_constraints": len(self.constraints),
        }

    def estimate_tokens(self) -> int:
        """Estimate total token count across all tiers (chars / 4)."""
        total_chars = len(self.to_sacred_block())
        if self.history_summary:
            total_chars += len(self.history_summary)
        for turn in self.recent_messages:
            total_chars += len(turn.content)
        return total_chars // CHARS_PER_TOKEN

    def silence_duration(self) -> float:
        """Seconds since last turn."""
        if not self.recent_messages:
            return 0.0
        return time.time() - self.recent_messages[-1].timestamp
