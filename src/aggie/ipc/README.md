# ipc - Inter-Process Communication Module

Handles communication between the daemon and control tools via Unix domain socket.

## Components

### `protocol.py` - Message Definitions

Defines the JSON protocol for daemon control.

**Commands:**
```python
class CommandType(Enum):
    MUTE = "mute"        # Stop listening
    UNMUTE = "unmute"    # Resume listening
    CANCEL = "cancel"    # Cancel current operation
    STATUS = "status"    # Get daemon status
    SHUTDOWN = "shutdown" # Stop the daemon
```

**Responses:**
- `SimpleResponse` - For mute/unmute/cancel/shutdown
- `StatusResponse` - For status command (includes state, muted, uptime)

**Wire format:**
```json
{"type": "status"}
{"status": "ok", "state": "IDLE", "muted": false, "uptime_seconds": 123.4}
```

### `server.py` - IPCServer

Unix domain socket server for receiving commands.

**Key details:**
- Socket path: `/run/user/{uid}/aggie.sock`
- Permissions: 0600 (owner only)
- Single-line JSON messages
- 5 second timeout per connection

**Usage:**
```python
async def handle_command(cmd: Command) -> Response:
    if cmd.type == CommandType.STATUS:
        return StatusResponse(...)
    ...

server = IPCServer(command_handler=handle_command)
await server.start()
# ... daemon runs ...
await server.stop()
```

## Adding a New Command

1. Add to `CommandType` enum in `protocol.py`
2. Handle in `daemon.py` `_handle_command()` method
3. Add subparser in `cli/ctl.py`

Example:
```python
# protocol.py
class CommandType(Enum):
    ...
    VOLUME = "volume"

# daemon.py
elif command.type == CommandType.VOLUME:
    # handle volume command
    return SimpleResponse(status=ResponseStatus.OK, message="Volume set")

# cli/ctl.py
subparsers.add_parser("volume", help="Set volume level")
```

## Security

- Socket is created with mode 0600 (owner read/write only)
- No authentication (relies on filesystem permissions)
- No encryption (local-only communication)

## Configuration

```yaml
ipc:
  socket_path: null  # Default: /run/user/{uid}/aggie.sock
```
