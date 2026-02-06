"""Haiku-powered sacred extraction and context compression."""

import asyncio
import json
import logging
import os
import re
from typing import Optional

from .project_state import (
    AgentDef,
    Constraint,
    Decision,
    ProjectState,
    ToolSchema,
)

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
Analyze the following conversation and extract any of these types of sacred content as JSON.
Deduplicate against the existing sacred content provided below.

Existing sacred content (do NOT re-extract these):
{existing_sacred}

Return ONLY a JSON object with these keys (omit keys with empty arrays):
{{
  "decisions": [{{"summary": "...", "rationale": "..."}}],
  "agents": [{{"name": "...", "role": "..."}}],
  "tools": [{{"name": "...", "description": "..."}}],
  "constraints": [{{"description": "..."}}]
}}

Conversation:
{conversation}
"""

COMPRESSION_PROMPT = """\
Summarize the following conversation history into a concise narrative paragraph.
Preserve key facts, decisions, and context needed to continue the conversation.
Do NOT include greetings or filler. Be factual and dense.

{conversation}
"""


class ContextCompressor:
    """Manages compression of working context via Haiku API calls.

    Triggers on turn count intervals. Uses fire-and-forget asyncio tasks.
    Handles race conditions by snapshotting before compression.
    """

    def __init__(
        self,
        state: ProjectState,
        api_key: Optional[str] = None,
        compress_every: int = 7,
        haiku_model: str = "claude-haiku-4-5-20251001",
        recent_window: int = 6,
    ) -> None:
        self._state = state
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._compress_every = compress_every
        self._haiku_model = haiku_model
        self._recent_window = recent_window
        self._lock = asyncio.Lock()
        self._client = None  # Lazy-loaded

    def _ensure_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    def should_compress(self) -> bool:
        """True when turn_count hits the compress interval and enough messages exist."""
        tc = self._state.turn_count
        return (
            tc > 0
            and tc % self._compress_every == 0
            and len(self._state.recent_messages) > self._recent_window
        )

    def maybe_trigger_compression(self) -> Optional[asyncio.Task]:
        """Fire-and-forget compression if conditions met. Returns the task or None."""
        if not self.should_compress():
            return None
        task = asyncio.create_task(self._run_compression())
        task.add_done_callback(self._compression_done)
        logger.info("Compression task triggered (fire-and-forget)")
        return task

    @staticmethod
    def _compression_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Compression task failed: {exc}", exc_info=exc)

    async def _run_compression(self) -> None:
        """Full compression pipeline: extract sacred → compress working."""
        if self._lock.locked():
            logger.debug("Compression already in progress, skipping")
            return

        async with self._lock:
            # Snapshot recent_messages length to handle race conditions
            snapshot_len = len(self._state.recent_messages)
            if snapshot_len <= self._recent_window:
                return

            messages_to_compress = list(self._state.recent_messages[:snapshot_len])

            # Build conversation text for prompts
            conversation = self._turns_to_text(messages_to_compress)

            # Step 1: Extract sacred content
            try:
                existing = self._state.to_sacred_block() or "(none)"
                extraction_prompt = EXTRACTION_PROMPT.format(
                    existing_sacred=existing,
                    conversation=conversation,
                )
                extraction_response = await self._call_haiku(extraction_prompt)
                self._promote_sacred(extraction_response)
            except Exception as e:
                logger.warning(f"Sacred extraction failed: {e}")

            # Step 2: Compress older messages into history summary
            try:
                # Include existing summary for continuity
                compress_input = ""
                if self._state.history_summary:
                    compress_input += f"Previous summary: {self._state.history_summary}\n\n"

                # Compress all but the recent window
                older = messages_to_compress[:-self._recent_window]
                if older:
                    compress_input += self._turns_to_text(older)
                    compression_prompt = COMPRESSION_PROMPT.format(
                        conversation=compress_input,
                    )
                    summary = await self._call_haiku(compression_prompt)
                    self._state.history_summary = summary.strip()
                    logger.info(
                        f"Compressed {len(older)} turns into summary "
                        f"({len(self._state.history_summary)} chars)"
                    )
            except Exception as e:
                logger.warning(f"Context compression failed: {e}")

            # Step 3: Trim recent_messages — only remove messages that existed
            # at snapshot time, preserving any added during compression
            new_messages_during = self._state.recent_messages[snapshot_len:]
            kept = self._state.recent_messages[
                snapshot_len - self._recent_window : snapshot_len
            ]
            self._state.recent_messages = kept + new_messages_during
            logger.debug(
                f"Trimmed to {len(self._state.recent_messages)} recent messages "
                f"({len(new_messages_during)} added during compression)"
            )

    def _promote_sacred(self, text: str) -> None:
        """Parse JSON from Haiku response and deduplicate into sacred tier."""
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse sacred extraction JSON: {text[:200]}")
            return

        # Deduplicate decisions
        existing_summaries = {d.summary for d in self._state.decisions}
        for d in data.get("decisions", []):
            if isinstance(d, dict) and d.get("summary") and d["summary"] not in existing_summaries:
                self._state.decisions.append(
                    Decision(summary=d["summary"], rationale=d.get("rationale", ""))
                )
                existing_summaries.add(d["summary"])

        # Deduplicate agents
        existing_agents = {a.name for a in self._state.agents}
        for a in data.get("agents", []):
            if isinstance(a, dict) and a.get("name") and a["name"] not in existing_agents:
                self._state.agents.append(
                    AgentDef(name=a["name"], role=a.get("role", ""))
                )
                existing_agents.add(a["name"])

        # Deduplicate tools
        existing_tools = {t.name for t in self._state.mcp_tools}
        for t in data.get("tools", []):
            if isinstance(t, dict) and t.get("name") and t["name"] not in existing_tools:
                self._state.mcp_tools.append(
                    ToolSchema(name=t["name"], description=t.get("description", ""))
                )
                existing_tools.add(t["name"])

        # Deduplicate constraints
        existing_constraints = {c.description for c in self._state.constraints}
        for c in data.get("constraints", []):
            if isinstance(c, dict) and c.get("description") and c["description"] not in existing_constraints:
                self._state.constraints.append(
                    Constraint(description=c["description"])
                )
                existing_constraints.add(c["description"])

        logger.debug(
            f"Sacred state: {len(self._state.decisions)} decisions, "
            f"{len(self._state.agents)} agents, "
            f"{len(self._state.mcp_tools)} tools, "
            f"{len(self._state.constraints)} constraints"
        )

    async def _call_haiku(self, prompt: str) -> str:
        """Make a single Haiku API call."""
        client = self._ensure_client()
        message = await client.messages.create(
            model=self._haiku_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    @staticmethod
    def _turns_to_text(turns: list) -> str:
        parts = []
        for t in turns:
            speaker = "User" if t.role == "user" else "Assistant"
            parts.append(f"{speaker}: {t.content}")
        return "\n".join(parts)
