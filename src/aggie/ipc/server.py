"""Unix domain socket server for daemon control."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .protocol import Command, ResponseStatus, SimpleResponse, Response

logger = logging.getLogger(__name__)

# Type alias for command handler
CommandHandler = Callable[[Command], Awaitable[Response]]


class IPCServer:
    """Unix domain socket server for daemon control.

    Accepts JSON commands over a Unix socket and returns JSON responses.
    Socket permissions are set to owner-only for security.
    """

    DEFAULT_SOCKET_PATH = "/run/user/{uid}/aggie.sock"

    def __init__(
        self,
        socket_path: Optional[str] = None,
        command_handler: Optional[CommandHandler] = None,
    ) -> None:
        """Initialize IPC server.

        Args:
            socket_path: Path to Unix socket. Defaults to /run/user/{uid}/aggie.sock.
            command_handler: Async function to handle commands.
        """
        if socket_path is None:
            socket_path = self.DEFAULT_SOCKET_PATH.format(uid=os.getuid())

        self._socket_path = Path(socket_path)
        self._command_handler = command_handler
        self._server: Optional[asyncio.Server] = None

    @property
    def socket_path(self) -> Path:
        """Path to the Unix socket."""
        return self._socket_path

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        try:
            # Read command (single line JSON)
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not data:
                return

            command = Command.from_json(data.decode().strip())
            logger.debug(f"Received command: {command.type.value}")

            # Process command
            if self._command_handler:
                response = await self._command_handler(command)
            else:
                response = SimpleResponse(
                    status=ResponseStatus.ERROR,
                    message="No handler configured",
                    error="Internal error",
                )

            # Send response
            writer.write(response.to_json().encode() + b"\n")
            await writer.drain()

        except asyncio.TimeoutError:
            logger.warning("Client connection timed out")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            try:
                response = SimpleResponse(
                    status=ResponseStatus.ERROR,
                    message="Command failed",
                    error=str(e),
                )
                writer.write(response.to_json().encode() + b"\n")
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        """Start the IPC server."""
        # Remove stale socket file
        if self._socket_path.exists():
            self._socket_path.unlink()

        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )

        # Set socket permissions (owner only)
        os.chmod(self._socket_path, 0o600)

        logger.info(f"IPC server listening on {self._socket_path}")

    async def stop(self) -> None:
        """Stop the IPC server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Clean up socket file
        if self._socket_path.exists():
            self._socket_path.unlink()

        logger.info("IPC server stopped")

    def set_handler(self, handler: CommandHandler) -> None:
        """Set the command handler.

        Args:
            handler: Async function to handle commands.
        """
        self._command_handler = handler
