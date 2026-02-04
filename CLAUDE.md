# CLAUDE.md - AI Assistant Context for AGGIE

## Project Overview

AGGIE is a privacy-first, always-on voice assistant daemon for Linux. It listens for a wake word, captures speech, transcribes locally, sends only text to Claude API, and speaks the response using local TTS.

## Key Architecture Decisions

### Privacy Model (Non-Negotiable)
- Raw audio NEVER leaves the device
- Only text transcripts are sent to Claude API
- All audio processing (wake word, VAD, STT, TTS) is local

### State Machine
```
IDLE → LISTENING → THINKING → SPEAKING → IDLE
```
- `IDLE`: Listening for wake word only (low resource)
- `LISTENING`: Recording user speech after wake word detected
- `THINKING`: Running STT → LLM → TTS pipeline
- `SPEAKING`: Playing audio response
- `MUTED`: All processing paused

### Threading Model
- Main asyncio event loop handles all coordination
- Audio capture uses sounddevice callbacks in a separate thread
- Thread-safe queue bridges audio thread to asyncio
- Heavy models (Whisper, Piper) load on-demand

## Code Organization

```
src/aggie/
├── daemon.py      # Main orchestrator - START HERE for understanding flow
├── state.py       # State machine - defines all valid transitions
├── config.py      # Configuration dataclasses
├── audio/         # Microphone capture, speaker playback, ring buffer
├── detection/     # Wake word detection (openWakeWord + Silero VAD)
├── stt/           # Speech-to-text (faster-whisper)
├── llm/           # Claude API client
├── tts/           # Text-to-speech (Piper)
├── ipc/           # Unix socket server for control commands
└── cli/           # aggie-ctl command-line tool
```

## Common Tasks

### Adding a new IPC command
1. Add command type to `ipc/protocol.py` `CommandType` enum
2. Handle it in `daemon.py` `_handle_command()` method
3. Add subparser in `cli/ctl.py`

### Changing audio processing
- Audio flows: `capture.py` → `daemon.py._process_audio_frame()` → `wakeword.py`
- Frame size is 80ms (1280 samples at 16kHz) - optimal for openWakeWord
- Ring buffer keeps ~1.5s pre-roll for capturing speech before wake word

### Modifying the response pipeline
- Pipeline in `daemon.py._process_recording()`: STT → LLM → TTS → Playback
- Each component is lazy-loaded via `_ensure_*()` methods

## Dependencies

Key packages and why they're used:
- `openwakeword`: Local wake word detection with built-in Silero VAD
- `faster-whisper`: Fast local STT (2x faster than whisper.cpp on CPU)
- `anthropic`: Claude API client
- `piper-tts`: Local neural TTS
- `sounddevice`: Cross-platform audio I/O with asyncio-friendly callbacks
- `soundfile`: Reading/writing audio files

## Configuration

Config loaded from `~/.config/aggie/config.yaml` (XDG compliant).
Defaults are sensible - most users only need to set `ANTHROPIC_API_KEY`.

## Testing

```bash
pytest                      # Run all tests
pytest -xvs tests/test_X.py # Run specific test with output
aggie --debug               # Run daemon with debug logging
aggie-ctl status            # Check daemon state
```

## Gotchas

1. **Audio format**: Everything expects 16kHz mono int16 PCM
2. **Wake word models**: Pre-trained models are CC BY-NC-SA (non-commercial)
3. **Piper voices**: Must be downloaded separately, check `~/.local/share/piper-voices/`
4. **Socket permissions**: IPC socket is chmod 600 (owner only)
5. **systemd**: Uses user services (`systemctl --user`), not system services
