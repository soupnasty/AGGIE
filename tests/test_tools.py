"""Tests for the tool access module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aggie.tools.bash import BASH_TOOL, check_command, execute_bash
from aggie.tools.registry import ToolRegistry


# --- check_command ---


class TestCheckCommand:
    """Tests for command safety checking."""

    def test_blocked_rm_rf_root(self):
        assert check_command("rm -rf /") == "block"

    def test_blocked_rm_rf_home(self):
        assert check_command("rm -rf ~") == "block"

    def test_blocked_mkfs(self):
        assert check_command("mkfs.ext4 /dev/sda1") == "block"

    def test_blocked_dd(self):
        assert check_command("dd if=/dev/zero of=/dev/sda") == "block"

    def test_blocked_fork_bomb(self):
        assert check_command(":(){ :|:& };:") == "block"

    def test_blocked_case_insensitive(self):
        assert check_command("RM -RF /") == "block"

    def test_warn_rm(self):
        assert check_command("rm foo.txt") == "warn"

    def test_warn_sudo(self):
        assert check_command("sudo apt update") == "warn"

    def test_warn_git_push(self):
        assert check_command("git push origin main") == "warn"

    def test_warn_pip_install(self):
        assert check_command("pip install requests") == "warn"

    def test_allow_ls(self):
        assert check_command("ls -la") == "allow"

    def test_allow_cat(self):
        assert check_command("cat README.md") == "allow"

    def test_allow_echo(self):
        assert check_command("echo hello") == "allow"

    def test_allow_git_status(self):
        assert check_command("git status") == "allow"

    def test_allow_python(self):
        assert check_command("python -c 'print(1)'") == "allow"


# --- execute_bash ---


class TestExecuteBash:
    """Tests for async bash execution."""

    @pytest.mark.asyncio
    async def test_echo_output(self, tmp_path):
        result = await execute_bash("echo hello", working_dir=str(tmp_path))
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, tmp_path):
        result = await execute_bash("exit 42", working_dir=str(tmp_path))
        assert "EXIT CODE: 42" in result

    @pytest.mark.asyncio
    async def test_stderr_captured(self, tmp_path):
        result = await execute_bash(
            "echo err >&2", working_dir=str(tmp_path)
        )
        assert "STDERR:" in result
        assert "err" in result

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        result = await execute_bash(
            "sleep 10", working_dir=str(tmp_path), timeout=0.5
        )
        assert "TIMEOUT" in result

    @pytest.mark.asyncio
    async def test_empty_output(self, tmp_path):
        result = await execute_bash("true", working_dir=str(tmp_path))
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_truncation(self, tmp_path):
        # Generate output larger than the limit
        result = await execute_bash(
            "seq 1 10000", working_dir=str(tmp_path), max_output_chars=200
        )
        assert "truncated" in result
        assert len(result) < 500  # Should be roughly 200 + truncation message


# --- ToolRegistry ---


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_get_tools(self):
        registry = ToolRegistry()
        tools = registry.get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_execute_echo(self, tmp_path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = await registry.execute("bash", {"command": "echo test"})
        assert result == "test"

    @pytest.mark.asyncio
    async def test_execute_blocked(self, tmp_path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = await registry.execute("bash", {"command": "rm -rf /"})
        assert "BLOCKED" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, tmp_path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = await registry.execute("unknown", {"command": "echo hi"})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_empty_command(self, tmp_path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = await registry.execute("bash", {"command": ""})
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_tool_log(self, tmp_path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        await registry.execute("bash", {"command": "echo logged"})
        assert len(registry._tool_log) == 1
        entry = registry._tool_log[0]
        assert entry["command"] == "echo logged"
        assert entry["category"] == "other"
        assert entry["policy"] == "allow"
        assert entry["duration_ms"] >= 0

    def test_categorize(self):
        assert ToolRegistry._categorize("git status") == "git"
        assert ToolRegistry._categorize("ls -la") == "filesystem"
        assert ToolRegistry._categorize("grep foo bar") == "search"
        assert ToolRegistry._categorize("python script.py") == "code_execution"
        assert ToolRegistry._categorize("curl https://example.com") == "network"
        assert ToolRegistry._categorize("sqlite3 db.sqlite") == "database"
        assert ToolRegistry._categorize("whoami") == "other"


# --- BASH_TOOL definition ---


class TestBashToolDefinition:
    """Tests for the tool definition schema."""

    def test_tool_has_name(self):
        assert BASH_TOOL["name"] == "bash"

    def test_tool_has_input_schema(self):
        schema = BASH_TOOL["input_schema"]
        assert schema["type"] == "object"
        assert "command" in schema["properties"]
        assert "command" in schema["required"]
