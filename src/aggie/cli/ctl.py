"""Command-line interface for controlling the AGGIE daemon."""

import argparse
import asyncio
import os
import sys

from aggie.ipc.protocol import (
    Command,
    CommandType,
    ContextStatusResponse,
    DebugDumpResponse,
    SimpleResponse,
    StatusResponse,
)


DEFAULT_SOCKET_PATH = f"/run/user/{os.getuid()}/aggie.sock"


async def send_command(socket_path: str, command: Command) -> str:
    """Send a command to the daemon and return the response.

    Args:
        socket_path: Path to daemon socket.
        command: Command to send.

    Returns:
        JSON response string.

    Raises:
        SystemExit: On connection errors.
    """
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)

        # Send command
        writer.write(command.to_json().encode() + b"\n")
        await writer.drain()

        # Read response
        data = await asyncio.wait_for(reader.readline(), timeout=5.0)

        writer.close()
        await writer.wait_closed()

        return data.decode().strip()

    except FileNotFoundError:
        print("Error: Daemon is not running (socket not found)", file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print("Error: Daemon refused connection", file=sys.stderr)
        sys.exit(1)
    except asyncio.TimeoutError:
        print("Error: Daemon did not respond in time", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for aggie-ctl."""
    parser = argparse.ArgumentParser(
        prog="aggie-ctl",
        description="Control the AGGIE voice assistant daemon",
    )
    parser.add_argument(
        "--socket",
        "-s",
        default=DEFAULT_SOCKET_PATH,
        help="Path to daemon socket",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Commands
    subparsers.add_parser("mute", help="Mute the assistant (stop listening)")
    subparsers.add_parser("unmute", help="Unmute the assistant (resume listening)")
    subparsers.add_parser("cancel", help="Cancel current operation")
    subparsers.add_parser("status", help="Get daemon status")
    subparsers.add_parser("shutdown", help="Shutdown the daemon")
    debug_parser = subparsers.add_parser("debug-dump", help="Dump debug logs for troubleshooting")
    debug_parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON lines (for piping to tools)",
    )
    context_parser = subparsers.add_parser("context", help="Show or manage session context")
    context_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the session context",
    )

    args = parser.parse_args()

    # Determine command type
    if args.command == "context":
        if args.clear:
            cmd_type = CommandType.CONTEXT_CLEAR
        else:
            cmd_type = CommandType.CONTEXT_STATUS
    else:
        command_map = {
            "mute": CommandType.MUTE,
            "unmute": CommandType.UNMUTE,
            "cancel": CommandType.CANCEL,
            "status": CommandType.STATUS,
            "shutdown": CommandType.SHUTDOWN,
            "debug-dump": CommandType.DEBUG_DUMP,
        }
        cmd_type = command_map[args.command]

    command = Command(type=cmd_type)
    response_json = asyncio.run(send_command(args.socket, command))

    # Parse and display response
    if args.command == "status":
        response = StatusResponse.from_json(response_json)
        print(f"State:  {response.state}")
        print(f"Muted:  {response.muted}")
        print(f"Uptime: {response.uptime_seconds:.1f}s")
        print(f"GPU:    {response.gpu or 'None (CPU mode)'}")
        if response.error:
            print(f"Error:  {response.error}")
    elif args.command == "debug-dump":
        response = DebugDumpResponse.from_json(response_json)
        if response.status.value == "ok":
            if args.raw:
                # Raw output for piping to jq/grep
                print(response.content, end="")
            else:
                print(f"# Debug log: {response.log_path} ({response.lines} lines)")
                print(f"# Tip: Use --raw | jq for JSON parsing")
                print()
                print(response.content, end="")
        else:
            print(f"Error: {response.error}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "context" and not args.clear:
        response = ContextStatusResponse.from_json(response_json)
        if response.status.value == "ok":
            print(f"Turns:    {response.turn_count}")
            print(f"Tokens:   ~{response.token_estimate}")
            print(f"Silence:  {response.silence_seconds:.1f}s")
            print(f"Summary:  {'Yes' if response.has_summary else 'No'}")
        else:
            print(f"Error: {response.error}", file=sys.stderr)
            sys.exit(1)
    else:
        response = SimpleResponse.from_json(response_json)
        if response.status.value == "ok":
            print(response.message)
        else:
            print(f"Error: {response.error}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
