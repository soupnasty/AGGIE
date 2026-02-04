# AGGIE — Desk-Only Wake-Word Voice Assistant (Privacy-First)

## What this project is
**AGGIE** is an always-on, desk-only voice assistant designed to run on a Linux workstation and be used hands-free with a simple wake word:

> "Hey Jarvis…" (customizable)

It sits idle most of the time, listens for the wake word, captures a single utterance, responds out loud, and then returns to idle.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Build and run
docker compose up --build

# Control (in another terminal)
docker compose exec aggie aggie-ctl status
docker compose exec aggie aggie-ctl mute
docker compose exec aggie aggie-ctl cancel

# Stop
docker compose down
```

### Option 2: Native Installation

**Prerequisites:**
```bash
# Ubuntu/Debian
sudo apt install python3-dev python3-venv portaudio19-dev libsndfile1

# Fedora
sudo dnf install python3-devel portaudio-devel libsndfile

# Arch
sudo pacman -S python portaudio libsndfile
```

**Install and run:**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install AGGIE
pip install -e .

# Set up configuration
mkdir -p ~/.config/aggie
cp config/aggie.yaml.example ~/.config/aggie/config.yaml

# Set your API key
export ANTHROPIC_API_KEY='your-key-here'

# Run
aggie --debug
```

### systemd Service (Native only)

```bash
cp config/systemd/aggie.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start aggie
systemctl --user enable aggie  # start on login
```

### Control Commands

```bash
aggie-ctl status    # Show current state
aggie-ctl mute      # Stop listening
aggie-ctl unmute    # Resume listening
aggie-ctl cancel    # Cancel current operation
aggie-ctl shutdown  # Stop the daemon
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         aggie-daemon                             │
│                                                                  │
│  ┌──────────┐   ┌─────────────┐   ┌───────────────┐             │
│  │Microphone│──▶│openWakeWord │──▶│faster-whisper │             │
│  │(capture) │   │+ Silero VAD │   │    (STT)      │             │
│  └──────────┘   └─────────────┘   └───────┬───────┘             │
│                                           │                      │
│                                   [transcript text only]         │
│                                           ▼                      │
│  ┌──────────┐   ┌─────────────┐   ┌───────────────┐             │
│  │ Speaker  │◀──│  Piper TTS  │◀──│  Claude API   │             │
│  │(playback)│   │   (local)   │   │  (reasoning)  │             │
│  └──────────┘   └─────────────┘   └───────────────┘             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Unix Socket IPC (/run/user/{uid}/aggie.sock)    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### State Machine

```
IDLE ──[wake word]──▶ LISTENING ──[silence]──▶ THINKING ──▶ SPEAKING ──▶ IDLE
  ▲                        │                       │             │
  └────────────────────────┴───────[cancel]────────┴─────────────┘

Any state ──[mute]──▶ MUTED ──[unmute]──▶ IDLE
```

## Project Structure

```
aggie/
├── pyproject.toml                 # Package configuration
├── config/
│   ├── aggie.yaml.example         # Example configuration
│   └── systemd/aggie.service      # systemd user service
├── src/aggie/
│   ├── __main__.py                # Entry point
│   ├── daemon.py                  # Main orchestrator
│   ├── state.py                   # State machine
│   ├── config.py                  # Configuration loading
│   ├── audio/
│   │   ├── capture.py             # Microphone input
│   │   ├── playback.py            # Speaker output
│   │   └── buffer.py              # Pre-roll ring buffer
│   ├── detection/
│   │   └── wakeword.py            # openWakeWord + VAD
│   ├── stt/
│   │   └── whisper.py             # faster-whisper
│   ├── llm/
│   │   └── claude.py              # Claude API client
│   ├── tts/
│   │   └── piper.py               # Piper TTS
│   ├── ipc/
│   │   ├── protocol.py            # JSON messages
│   │   └── server.py              # Unix socket server
│   └── cli/
│       └── ctl.py                 # aggie-ctl command
└── tests/
```

## Configuration

Configuration is loaded from `~/.config/aggie/config.yaml`. See `config/aggie.yaml.example` for all options.

Key settings:

```yaml
wakeword:
  model: "hey_jarvis"      # Wake word model
  threshold: 0.5           # Detection sensitivity (0-1)

stt:
  model_size: "small"      # tiny, base, small, medium, large-v3
  device: "auto"           # auto, cpu, cuda

llm:
  model: "claude-sonnet-4-20250514"
  max_tokens: 300

tts:
  voice_model: "en_US-lessac-medium"
  speaking_rate: 1.0
```

## Privacy Model (Non-Negotiables)

AGGIE is built around a privacy-first pipeline:

- **Raw microphone audio stays on-device.** All audio processing (wake word, VAD, STT, TTS) happens locally.
- The system only transmits **text transcripts** to external services (Claude API).
- Users have clear controls to **mute**, **cancel**, and understand what the assistant is doing at any moment.

## Components

| Component | Technology | Location |
|-----------|------------|----------|
| Wake Word | [openWakeWord](https://github.com/dscripka/openWakeWord) + Silero VAD | Local |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Local |
| Reasoning | [Claude API](https://www.anthropic.com/api) | Remote (text only) |
| Text-to-Speech | [Piper](https://github.com/rhasspy/piper) | Local |
| Service Manager | systemd | Local |
| IPC | Unix domain socket (JSON) | Local |

## Core Goals

- **Always on, low friction**: leave it running continuously and speak to it occasionally.
- **Wake-word first interaction**: no push-to-talk required for normal use.
- **Desk-only tuning**: optimized for close-range speech with reduced room pickup and fewer false triggers.
- **Clear, explicit states**: *Idle → Listening → Thinking → Speaking*.
- **Fast and reliable**: predictable response flow with safe failure behavior.

## User Experience

- **Primary interaction**: wake word → speak → response out loud.
- **Controls**
  - **Mute / unmute** (stop listening while muted)
  - **Stop / cancel** (interrupt speaking and abort the current request)
- **Operational stability**
  - Background daemon that stays running reliably
  - CLI control tool for status and commands
  - Can be managed via systemd

## Project Values

- **Privacy by default**
- **Modularity and replaceability**
- **Reliability over novelty**
- **Simple, understandable UX**

## License

GPL-3.0-or-later (to comply with Piper TTS licensing)
