"""Prompt assembly from ProjectState tiers."""

import logging
from typing import Optional

from .project_state import ProjectState

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """You are AGGIE, a helpful voice assistant.
Keep responses concise and conversational - they will be spoken aloud.
Aim for 1-3 sentences unless the user asks for more detail.
Do not use markdown formatting, bullet points, or special characters.
Respond naturally as if speaking to someone."""


class PromptComposer:
    """Assembles system prompt + messages from ProjectState.

    System prompt = base identity + sacred block (stable prefix for caching).
    Messages = optional history summary pair + recent messages.
    """

    def __init__(
        self,
        state: ProjectState,
        base_prompt: Optional[str] = None,
    ) -> None:
        self._state = state
        self._base_prompt = base_prompt or BASE_SYSTEM_PROMPT

    def compose(self) -> tuple[str, list[dict]]:
        """Build (system_prompt, messages) for the Claude API.

        Returns:
            Tuple of (system_prompt, messages list).
        """
        system_prompt = self._build_system_prompt()
        messages = self._build_messages()
        return system_prompt, messages

    def _build_system_prompt(self) -> str:
        sacred = self._state.to_sacred_block()
        if sacred:
            return self._base_prompt + "\n\n" + sacred
        return self._base_prompt

    def _build_messages(self) -> list[dict]:
        messages: list[dict] = []

        # History summary as synthetic user/assistant pair
        if self._state.history_summary:
            messages.append({
                "role": "user",
                "content": f"[Previous conversation summary: {self._state.history_summary}]",
            })
            messages.append({
                "role": "assistant",
                "content": "I understand. I'll keep that context in mind.",
            })

        # Recent messages
        for turn in self._state.recent_messages:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages
