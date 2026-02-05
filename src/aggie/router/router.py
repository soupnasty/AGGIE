"""Query router for AGGIE voice assistant.

Routes queries to the appropriate backend:
- System commands (time, date) - no LLM needed
- Llama (local) - default for most queries
- Claude (API) - explicit trigger phrases
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

from .llama import LlamaClient, LlamaError
from .system_commands import SystemResponse, check_system_command

logger = logging.getLogger(__name__)


class ResponseSource(Enum):
    """Source of the response."""

    SYSTEM = "system"  # Local system command (time, date)
    LLAMA = "llama"  # Local Llama model via Ollama
    CLAUDE = "claude"  # Claude API


@dataclass
class RouteDecision:
    """Result of routing decision."""

    source: ResponseSource
    query: str  # Possibly modified query (trigger phrase removed)
    system_response: Optional[SystemResponse] = None  # For SYSTEM source


# Claude trigger patterns
CLAUDE_TRIGGERS = [
    re.compile(r"^use claude\b", re.IGNORECASE),
    re.compile(r"^ask claude\b", re.IGNORECASE),
    re.compile(r"^claude,?\s+", re.IGNORECASE),
]


class Router:
    """Routes queries to appropriate LLM backend.

    Routing priority:
    1. System commands (time, date) - handled locally
    2. Claude triggers ("use claude", "ask claude") - sent to Claude API
    3. Default - sent to local Llama model
    """

    def __init__(
        self,
        llama_model: str = "llama3.1:8b",
        llama_host: str = "http://localhost:11434",
        claude_fallback: bool = True,
    ) -> None:
        """Initialize router.

        Args:
            llama_model: Ollama model name for local inference.
            llama_host: Ollama API host URL.
            claude_fallback: Fall back to Claude if Llama unavailable.
        """
        self._llama = LlamaClient(model=llama_model, host=llama_host)
        self._claude_fallback = claude_fallback
        self._llama_available: Optional[bool] = None

    async def close(self) -> None:
        """Clean up resources."""
        await self._llama.close()

    def route(self, query: str) -> RouteDecision:
        """Decide where to route a query.

        Args:
            query: User query text.

        Returns:
            RouteDecision indicating source and possibly modified query.
        """
        # 1. Check system commands first
        system_response = check_system_command(query)
        if system_response:
            return RouteDecision(
                source=ResponseSource.SYSTEM,
                query=query,
                system_response=system_response,
            )

        # 2. Check Claude triggers
        for pattern in CLAUDE_TRIGGERS:
            match = pattern.match(query)
            if match:
                # Remove trigger phrase from query
                clean_query = query[match.end():].strip()
                if not clean_query:
                    clean_query = query  # Keep original if nothing left
                logger.info(f"Claude trigger matched, routing to Claude")
                return RouteDecision(
                    source=ResponseSource.CLAUDE,
                    query=clean_query,
                )

        # 3. Default to Llama
        return RouteDecision(
            source=ResponseSource.LLAMA,
            query=query,
        )

    async def check_llama_available(self) -> bool:
        """Check if Llama/Ollama is available.

        Caches result to avoid repeated checks.
        """
        if self._llama_available is None:
            self._llama_available = await self._llama.is_available()
            if self._llama_available:
                logger.info("Llama is available via Ollama")
            else:
                logger.warning("Llama is not available")
        return self._llama_available

    async def get_llama_response(self, messages: list[dict]) -> str:
        """Get response from Llama.

        Args:
            messages: Conversation messages.

        Returns:
            Response text.

        Raises:
            LlamaError: On failure.
        """
        return await self._llama.get_response(messages)

    async def stream_llama_response(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """Stream response from Llama.

        Args:
            messages: Conversation messages.

        Yields:
            Response text chunks.

        Raises:
            LlamaError: On failure.
        """
        async for chunk in self._llama.stream_response(messages):
            yield chunk

    @property
    def llama_client(self) -> LlamaClient:
        """Access to underlying Llama client."""
        return self._llama

    @property
    def claude_fallback_enabled(self) -> bool:
        """Whether Claude fallback is enabled."""
        return self._claude_fallback
