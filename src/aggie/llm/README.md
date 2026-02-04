# llm - LLM Client Module

Handles communication with Claude API for response generation.

## Components

### `claude.py` - ClaudeClient

Async client for Claude API with voice-assistant-optimized prompting.

**Key details:**
- Async API calls (non-blocking)
- Single-shot requests (no conversation memory)
- System prompt optimized for spoken responses
- Only sends text - no audio ever leaves device

**Usage:**
```python
client = ClaudeClient(
    model="claude-sonnet-4-20250514",
    max_tokens=300,
)

response = await client.get_response("What's the weather like?")
# response is a string suitable for TTS
```

## System Prompt

The client uses a system prompt that instructs Claude to:
- Keep responses concise (1-3 sentences)
- Avoid markdown, bullet points, special characters
- Respond conversationally (will be spoken aloud)

## Configuration

```yaml
llm:
  model: "claude-sonnet-4-20250514"
  max_tokens: 300
```

API key is read from `ANTHROPIC_API_KEY` environment variable.

## Privacy Model

**Critical:** Only text transcripts are sent to Claude API.

```
Audio → [LOCAL: STT] → Text transcript → [REMOTE: Claude] → Response text → [LOCAL: TTS] → Audio
         ^^^^^^^^^^^                                                          ^^^^^^^^^^^^
         Never leaves                                                         Never leaves
         device                                                               device
```

## Error Handling

The client handles:
- Rate limiting (with backoff)
- Network errors
- API errors

Errors are logged and the daemon returns to IDLE state gracefully.

## Extending

To add conversation memory:
1. Store messages in a list
2. Pass full message history to `messages.create()`
3. Consider token limits and context window

Current design intentionally uses single-shot for simplicity and privacy.
