# cli - Command Line Interface Module

Command-line tools for controlling AGGIE.

## Components

### `ctl.py` - aggie-ctl

Control tool for the AGGIE daemon.

**Commands:**
```bash
aggie-ctl status    # Show daemon state, muted status, uptime
aggie-ctl mute      # Stop listening for wake word
aggie-ctl unmute    # Resume listening
aggie-ctl cancel    # Cancel current operation (stops TTS playback)
aggie-ctl shutdown  # Gracefully stop the daemon
```

**Options:**
```bash
aggie-ctl --socket /path/to/socket status  # Use custom socket path
```

## Usage Examples

```bash
# Check if daemon is running
aggie-ctl status
# Output:
# State:  IDLE
# Muted:  False
# Uptime: 3600.0s

# Mute before a meeting
aggie-ctl mute

# Resume after meeting
aggie-ctl unmute

# Stop a long response
aggie-ctl cancel

# Shutdown for maintenance
aggie-ctl shutdown
```

## Exit Codes

- `0` - Success
- `1` - Error (daemon not running, command failed, etc.)

## Error Messages

| Error | Meaning |
|-------|---------|
| "Daemon is not running (socket not found)" | Daemon isn't started or socket path is wrong |
| "Daemon refused connection" | Socket exists but daemon isn't accepting |
| "Daemon did not respond in time" | Daemon is stuck or overloaded |

## Adding Commands

1. Add `CommandType` in `ipc/protocol.py`
2. Add handler in `daemon.py`
3. Add subparser here:

```python
subparsers.add_parser("mycommand", help="Description of command")
command_map["mycommand"] = CommandType.MYCOMMAND
```
