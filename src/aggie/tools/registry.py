"""Tool registry for managing and routing tool calls."""

import datetime
import logging
import time
from typing import Optional

from .bash import BASH_TOOL, check_command, execute_bash

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages available tools and routes execution.

    Currently supports only the bash tool. Designed to support
    MCP servers in the future as usage patterns emerge.
    """

    def __init__(
        self,
        working_dir: str = "~",
        timeout: float = 30.0,
        max_output_chars: int = 16000,
    ) -> None:
        self._working_dir = working_dir
        self._timeout = timeout
        self._max_output_chars = max_output_chars
        self._tool_log: list[dict] = []

    def get_tools(self) -> list[dict]:
        """Return tool definitions for the Claude API."""
        return [BASH_TOOL]

    async def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call with safety checks and logging.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Tool input parameters.

        Returns:
            Tool output as a string.
        """
        if tool_name != "bash":
            return f"Unknown tool: {tool_name}"

        command = tool_input.get("command", "")
        if not command:
            return "ERROR: No command provided."

        policy = check_command(command)

        if policy == "block":
            logger.warning(f"BLOCKED command: {command}")
            self._log_call(command, "BLOCKED", 0, policy)
            return "BLOCKED: This command is not allowed for safety reasons."

        if policy == "warn":
            logger.warning(f"Executing warned command: {command}")

        start = time.monotonic()
        output = await execute_bash(
            command,
            working_dir=self._working_dir,
            timeout=self._timeout,
            max_output_chars=self._max_output_chars,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            f"Tool call: {command!r} ({duration_ms}ms, "
            f"{len(output)} chars, policy={policy})"
        )
        self._log_call(command, output, duration_ms, policy)

        return output

    def _log_call(
        self, command: str, output: str, duration_ms: int, policy: str
    ) -> None:
        """Log a tool call for future migration analysis."""
        self._tool_log.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "command": command,
                "output_chars": len(output),
                "duration_ms": duration_ms,
                "policy": policy,
                "category": self._categorize(command),
            }
        )

    @staticmethod
    def _categorize(command: str) -> str:
        """Auto-categorize a command for migration analysis."""
        cmd = command.strip().split()[0] if command.strip() else ""
        categories = {
            "cat": "filesystem",
            "ls": "filesystem",
            "find": "filesystem",
            "head": "filesystem",
            "tail": "filesystem",
            "wc": "filesystem",
            "mkdir": "filesystem",
            "touch": "filesystem",
            "cp": "filesystem",
            "mv": "filesystem",
            "git": "git",
            "python": "code_execution",
            "python3": "code_execution",
            "pytest": "code_execution",
            "grep": "search",
            "rg": "search",
            "ag": "search",
            "sqlite3": "database",
            "psql": "database",
            "curl": "network",
            "wget": "network",
        }
        return categories.get(cmd, "other")
