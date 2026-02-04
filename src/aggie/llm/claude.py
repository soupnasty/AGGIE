"""Claude API client for response generation."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Async client for Claude API.

    Handles single-shot requests (no conversation memory) as per
    AGGIE's design. Only text transcripts are sent - no audio.
    """

    SYSTEM_PROMPT = """You are AGGIE, a helpful voice assistant.
Keep responses concise and conversational - they will be spoken aloud.
Aim for 1-3 sentences unless the user asks for more detail.
Do not use markdown formatting, bullet points, or special characters.
Respond naturally as if speaking to someone."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 300,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                     ANTHROPIC_API_KEY environment variable.
            model: Claude model to use.
            max_tokens: Maximum tokens in response.

        Raises:
            ValueError: If no API key is available.
        """
        from anthropic import AsyncAnthropic

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Please set it or pass api_key parameter."
            )

        self._client = AsyncAnthropic(api_key=self._api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def get_response(self, transcript: str) -> str:
        """Get a response from Claude for the given transcript.

        Args:
            transcript: User's spoken text.

        Returns:
            Claude's response text.

        Raises:
            anthropic.APIError: On API errors.
        """
        if not transcript.strip():
            return ""

        logger.info(f"Sending to Claude: '{transcript[:80]}{'...' if len(transcript) > 80 else ''}'")

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": transcript,
                }
            ],
        )

        # Extract text from response
        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        logger.info(
            f"Claude response: '{response_text[:80]}{'...' if len(response_text) > 80 else ''}'"
        )
        return response_text
