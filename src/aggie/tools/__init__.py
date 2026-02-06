"""Tool access for AGGIE — shell-first, MCP-later."""

from .bash import BASH_TOOL, check_command, execute_bash
from .registry import ToolRegistry

__all__ = ["BASH_TOOL", "ToolRegistry", "check_command", "execute_bash"]
