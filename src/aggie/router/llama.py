"""Ollama/Llama client for local LLM inference."""

import asyncio
import logging
from typing import AsyncIterator, Optional, Union

logger = logging.getLogger(__name__)


class LlamaError(Exception):
    """Raised when Llama/Ollama request fails."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LlamaClient:
    """Async client for Ollama API.

    Provides streaming responses from locally-running Llama models.
    """

    SYSTEM_PROMPT = """You are AGGIE, a helpful voice assistant.
Keep responses concise and conversational - they will be spoken aloud.
Aim for 1-3 sentences unless the user asks for more detail.
Do not use markdown formatting, bullet points, or special characters.
Respond naturally as if speaking to someone."""

    def __init__(
        self,
        model: str = "llama3.1:8b-instruct-q4_0",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        """Initialize Llama client.

        Args:
            model: Ollama model name (e.g., "llama3.1:8b-instruct-q4_0").
            host: Ollama API host URL.
            timeout: Request timeout in seconds.
        """
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._client: Optional["httpx.AsyncClient"] = None

    async def _ensure_client(self) -> "httpx.AsyncClient":
        """Ensure HTTP client is initialized."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._host,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_response(
        self, input_data: Union[str, list[dict]]
    ) -> str:
        """Get a complete response from Llama.

        Args:
            input_data: Either a single query string or a list
                       of message dicts with 'role' and 'content' keys.

        Returns:
            Llama's response text.

        Raises:
            LlamaError: On API errors.
        """
        # Collect streaming response
        chunks = []
        async for chunk in self.stream_response(input_data):
            chunks.append(chunk)
        return "".join(chunks)

    async def stream_response(
        self, input_data: Union[str, list[dict]]
    ) -> AsyncIterator[str]:
        """Stream a response from Llama, yielding text chunks.

        Args:
            input_data: Either a single query string or a list
                       of message dicts with 'role' and 'content' keys.

        Yields:
            Text chunks as they arrive from the API.

        Raises:
            LlamaError: On API errors.
        """
        import httpx
        import json

        # Build messages list
        if isinstance(input_data, str):
            if not input_data.strip():
                return
            messages = [{"role": "user", "content": input_data}]
            log_preview = input_data[:80]
        else:
            if not input_data:
                return
            messages = input_data
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            log_preview = last_user[:80]

        # Prepend system message
        full_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *messages,
        ]

        logger.info(
            f"Streaming from Llama ({len(messages)} messages): "
            f"'{log_preview}{'...' if len(log_preview) >= 80 else ''}'"
        )

        client = await self._ensure_client()

        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": full_messages,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LlamaError(
                        f"Ollama API error {response.status_code}: {error_text.decode()}",
                        retryable=response.status_code >= 500,
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:
                                yield content

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse Ollama response: {line}")

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Ollama at {self._host}: {e}")
            raise LlamaError(
                f"Cannot connect to Ollama at {self._host}. Is it running?",
                retryable=True,
            ) from e

        except httpx.TimeoutException as e:
            logger.error(f"Ollama request timed out: {e}")
            raise LlamaError("Ollama request timed out", retryable=True) from e

        except Exception as e:
            if isinstance(e, LlamaError):
                raise
            logger.error(f"Unexpected error calling Ollama: {e}", exc_info=True)
            raise LlamaError(f"Unexpected error: {e}", retryable=False) from e

    async def is_available(self) -> bool:
        """Check if Ollama is running and the model is available.

        Returns:
            True if Ollama is reachable and model exists.
        """
        try:
            client = await self._ensure_client()
            response = await client.get("/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check if our model is available (handle tag variations)
            model_base = self._model.split(":")[0]
            for model in models:
                if model.startswith(model_base):
                    return True

            logger.warning(
                f"Model {self._model} not found. Available: {models}"
            )
            return False

        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False
