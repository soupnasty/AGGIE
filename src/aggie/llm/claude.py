"""Claude API client for response generation."""

import asyncio
import logging
import os
from typing import Optional, Union

import httpx

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Raised when Claude API request fails after retries."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


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

    # HTTP status codes that are retryable
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 300,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                     ANTHROPIC_API_KEY environment variable.
            model: Claude model to use.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Maximum retries for transient errors.

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

        self._client = AsyncAnthropic(
            api_key=self._api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def get_response(
        self, input_data: Union[str, list[dict]]
    ) -> str:
        """Get a response from Claude with retry logic.

        Args:
            input_data: Either a single transcript string (legacy) or a list
                       of message dicts with 'role' and 'content' keys.

        Returns:
            Claude's response text.

        Raises:
            APIError: On API errors after retries exhausted.
        """
        from anthropic import APIConnectionError, APIStatusError, APITimeoutError

        # Handle both string (legacy) and messages list
        if isinstance(input_data, str):
            if not input_data.strip():
                return ""
            messages = [{"role": "user", "content": input_data}]
            log_preview = input_data[:80]
        else:
            if not input_data:
                return ""
            messages = input_data
            # Preview the last user message for logging
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            log_preview = last_user[:80]

        logger.info(
            f"Sending to Claude ({len(messages)} messages): "
            f"'{log_preview}{'...' if len(log_preview) >= 80 else ''}'"
        )

        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                message = await self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self.SYSTEM_PROMPT,
                    messages=messages,
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

            except APITimeoutError as e:
                last_error = e
                logger.warning(f"Claude API timeout (attempt {attempt + 1}/{self._max_retries + 1})")

            except APIConnectionError as e:
                last_error = e
                logger.warning(f"Claude API connection error (attempt {attempt + 1}/{self._max_retries + 1}): {e}")

            except APIStatusError as e:
                last_error = e
                if e.status_code in self.RETRYABLE_STATUS_CODES:
                    logger.warning(
                        f"Claude API error {e.status_code} (attempt {attempt + 1}/{self._max_retries + 1}): {e.message}"
                    )
                else:
                    # Non-retryable error (e.g., 400, 401, 403)
                    logger.error(f"Claude API error {e.status_code}: {e.message}")
                    raise APIError(f"Claude API error: {e.message}", retryable=False) from e

            except Exception as e:
                # Unexpected error - don't retry
                logger.error(f"Unexpected error calling Claude API: {e}", exc_info=True)
                raise APIError(f"Unexpected error: {e}", retryable=False) from e

            # Exponential backoff before retry
            if attempt < self._max_retries:
                delay = min(2 ** attempt, 10)  # 1s, 2s, 4s, max 10s
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"Claude API failed after {self._max_retries + 1} attempts")
        raise APIError(
            f"Claude API failed after {self._max_retries + 1} attempts: {last_error}",
            retryable=True
        )
