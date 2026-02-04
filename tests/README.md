# tests - Test Suite

Test suite for AGGIE using pytest.

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_state.py

# Run specific test
pytest tests/test_state.py::test_valid_transitions

# Run with coverage
pytest --cov=aggie --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

## Test Organization

```
tests/
├── conftest.py          # Shared fixtures
├── test_state.py        # State machine tests
├── test_config.py       # Configuration tests
├── test_audio.py        # Audio module tests
├── test_protocol.py     # IPC protocol tests
└── test_integration.py  # End-to-end tests (slow)
```

## Fixtures

Defined in `conftest.py`:

| Fixture | Description |
|---------|-------------|
| `sample_audio` | 1 second of 440Hz sine wave (16kHz, int16) |
| `silence_audio` | 1 second of silence |
| `temp_config` | Temporary config file for testing |

## Writing Tests

### Unit Test Example
```python
import pytest
from aggie.state import State, StateMachine

@pytest.mark.asyncio
async def test_idle_to_listening():
    sm = StateMachine()
    assert sm.state == State.IDLE

    result = await sm.transition(State.LISTENING)

    assert result is True
    assert sm.state == State.LISTENING
```

### Integration Test Example
```python
import pytest

@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline():
    # This test requires actual audio hardware
    ...
```

## Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.asyncio` | Async test |
| `@pytest.mark.slow` | Slow test (skipped with `-m "not slow"`) |
| `@pytest.mark.integration` | Requires external services/hardware |

## Mocking

For tests that don't need real audio/models:

```python
from unittest.mock import Mock, AsyncMock, patch

@patch('aggie.stt.whisper.WhisperModel')
def test_stt_transcribe(mock_model):
    mock_model.return_value.transcribe.return_value = (
        [Mock(text="Hello world")],
        Mock()
    )
    # ... test code
```

## CI/CD

Tests run automatically on:
- Push to main
- Pull requests

Slow/integration tests are skipped in CI unless explicitly enabled.
