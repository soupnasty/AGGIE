"""MCP server lifecycle and tool routing."""

import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPManager:
    """Manages multiple MCP server connections.

    Starts servers, discovers tools, routes tool calls to the correct
    session, and shuts everything down cleanly.
    """

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        # tool_name -> (session, server_name)
        self._tool_map: dict[str, tuple[ClientSession, str]] = {}
        # tool_name -> tool definition dict (Anthropic API format)
        self._tool_defs: dict[str, dict] = {}

    async def start(self, servers: list) -> None:
        """Connect to all configured MCP servers.

        Args:
            servers: List of MCPServerConfig objects.
        """
        for server_config in servers:
            try:
                await self._connect_server(server_config)
            except Exception as e:
                logger.warning(
                    f"MCP server '{server_config.name}' failed to start: {e}. "
                    f"Continuing without it."
                )

        server_count = len(set(s for _, s in self._tool_map.values()))
        logger.info(
            f"MCP manager started: {len(self._tool_defs)} tools "
            f"from {server_count} servers"
        )

    async def _connect_server(self, server_config) -> None:
        """Connect to a single MCP server and register its tools."""
        params = StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            env=server_config.env if server_config.env else None,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        # Discover tools
        response = await session.list_tools()
        for tool in response.tools:
            tool_name = tool.name
            if tool_name in self._tool_map:
                tool_name = f"{server_config.name}__{tool.name}"
                logger.warning(
                    f"Tool name collision: '{tool.name}' already registered. "
                    f"Using '{tool_name}' for server '{server_config.name}'."
                )

            self._tool_map[tool_name] = (session, server_config.name)
            self._tool_defs[tool_name] = {
                "name": tool_name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }

        logger.info(
            f"MCP server '{server_config.name}': "
            f"{len(response.tools)} tools registered"
        )

    def get_tools(self) -> list[dict]:
        """Return all MCP tool definitions in Anthropic API format."""
        return list(self._tool_defs.values())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to an MCP server."""
        return tool_name in self._tool_map

    async def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute an MCP tool call.

        Args:
            tool_name: Name of the MCP tool.
            tool_input: Tool input arguments.

        Returns:
            Tool result as a string.
        """
        if tool_name not in self._tool_map:
            return f"Unknown MCP tool: {tool_name}"

        session, server_name = self._tool_map[tool_name]

        # Restore original name if it was prefixed due to collision
        original_name = tool_name
        if "__" in tool_name:
            original_name = tool_name.split("__", 1)[1]

        try:
            result = await session.call_tool(original_name, tool_input)
            return self._result_to_string(result)
        except Exception as e:
            logger.error(
                f"MCP tool '{tool_name}' (server '{server_name}') failed: {e}"
            )
            return f"ERROR: MCP tool execution failed: {e}"

    @staticmethod
    def _result_to_string(result) -> str:
        """Convert MCP CallToolResult to a plain string."""
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            elif hasattr(content, "type"):
                parts.append(f"[{content.type} content]")
            else:
                parts.append(str(content))

        return "\n".join(parts) if parts else "(no output)"

    async def shutdown(self) -> None:
        """Shut down all MCP server connections."""
        try:
            await self._exit_stack.aclose()
            logger.info("MCP manager shut down")
        except Exception as e:
            logger.warning(f"Error during MCP shutdown: {e}")
        self._tool_map.clear()
        self._tool_defs.clear()
