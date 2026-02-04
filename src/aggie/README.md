# aggie - Core Package

Main package for the AGGIE voice assistant daemon.

## Module Overview

| Module | Purpose |
|--------|---------|
| `daemon.py` | Main orchestrator - coordinates all components |
| `state.py` | State machine with async transitions |
| `config.py` | Configuration loading from YAML |
| `__main__.py` | Entry point for `aggie` command |

## Subpackages

| Package | Purpose |
|---------|---------|
| `audio/` | Microphone capture, speaker playback, audio buffering |
| `detection/` | Wake word detection with VAD |
| `stt/` | Speech-to-text transcription |
| `llm/` | LLM API client for response generation |
| `tts/` | Text-to-speech synthesis |
| `ipc/` | Unix socket server and protocol |
| `cli/` | Command-line control tool |

## Entry Points

Defined in `pyproject.toml`:
- `aggie` → `aggie.__main__:main` - Runs the daemon
- `aggie-ctl` → `aggie.cli.ctl:main` - Control tool

## Data Flow

```
Microphone → AudioCapture → WakeWordDetector → [wake word detected]
                                                      ↓
                                              Recording Buffer
                                                      ↓
                                              SpeechToText
                                                      ↓
                                              ClaudeClient (text only)
                                                      ↓
                                              TextToSpeech
                                                      ↓
                                              AudioPlayback → Speaker
```

## Key Classes

- `AggieDaemon` - Main daemon class, owns all components
- `StateMachine` - Manages state transitions with callbacks
- `Config` - Dataclass hierarchy for all configuration
