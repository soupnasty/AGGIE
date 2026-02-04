# config - Configuration Files

Configuration templates and service definitions for AGGIE.

## Files

### `aggie.yaml.example`

Example configuration file. Copy to `~/.config/aggie/config.yaml` and customize.

**Setup:**
```bash
mkdir -p ~/.config/aggie
cp aggie.yaml.example ~/.config/aggie/config.yaml
```

### `systemd/aggie.service`

systemd user service unit for running AGGIE as a background service.

**Setup:**
```bash
# Install service
mkdir -p ~/.config/systemd/user
cp systemd/aggie.service ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload

# Start service
systemctl --user start aggie

# Enable on login
systemctl --user enable aggie

# Check status
systemctl --user status aggie

# View logs
journalctl --user -u aggie -f
```

## Configuration Search Path

AGGIE looks for configuration in this order:
1. Path specified via `--config` flag
2. `$XDG_CONFIG_HOME/aggie/config.yaml` (usually `~/.config/aggie/config.yaml`)
3. `/etc/aggie/config.yaml` (system-wide)
4. Built-in defaults (if no config file found)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required. Claude API key. |
| `XDG_CONFIG_HOME` | Config directory (default: `~/.config`) |

**Tip:** Store API key in `~/.config/aggie/env` and reference in systemd service:
```bash
# ~/.config/aggie/env
ANTHROPIC_API_KEY=sk-ant-...
```

## Configuration Sections

| Section | Purpose |
|---------|---------|
| `audio` | Sample rate, silence detection, recording limits |
| `wakeword` | Wake word model, detection threshold |
| `stt` | Whisper model size, device, language |
| `llm` | Claude model, max tokens |
| `tts` | Piper voice, speaking rate |
| `ipc` | Socket path |
| `logging` | Log level, log file |

See `aggie.yaml.example` for all options with documentation.
