"""Bash tool for executing shell commands."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Tool definition for the Anthropic API
BASH_TOOL = {
    "name": "bash",
    "description": (
        "Run a bash command on the local machine. "
        "Use for reading files, writing files, running tests, "
        "git operations, searching codebases, and any other local task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}

# Commands that are never allowed
BLOCKED_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "> /dev/sd",
    "dd if=",
    ":(){ :|:& };:",
]

# Commands that get logged with a warning but still execute
WARN_PATTERNS = [
    "rm ",
    "sudo ",
    "git push",
    "git reset --hard",
    "pip install",
    "apt install",
    "chmod ",
    "chown ",
    "mv ",
]


def check_command(command: str) -> str:
    """Check a command against safety patterns.

    Args:
        command: The shell command to check.

    Returns:
        "allow", "warn", or "block".
    """
    cmd_lower = command.lower().strip()

    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return "block"

    for pattern in WARN_PATTERNS:
        if pattern in cmd_lower:
            return "warn"

    return "allow"


async def execute_bash(
    command: str,
    working_dir: str = "~",
    timeout: float = 30.0,
    max_output_chars: int = 16000,
) -> str:
    """Execute a bash command asynchronously.

    Args:
        command: The shell command to execute.
        working_dir: Working directory for the command.
        timeout: Maximum execution time in seconds.
        max_output_chars: Maximum characters in output before truncation.

    Returns:
        Command output (stdout + stderr) as a string.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"TIMEOUT: Command exceeded {timeout}s limit."

        output = stdout.decode(errors="replace")
        if stderr:
            stderr_text = stderr.decode(errors="replace")
            if stderr_text.strip():
                output += f"\nSTDERR:\n{stderr_text}"
        if proc.returncode != 0:
            output += f"\nEXIT CODE: {proc.returncode}"

        output = output.strip()

        if not output:
            return "(no output)"

        # Truncate from the middle if too long
        if len(output) > max_output_chars:
            half = max_output_chars // 2
            total_lines = output.count("\n") + 1
            output = (
                output[:half]
                + f"\n\n[... truncated — {total_lines} total lines ...]\n\n"
                + output[-half:]
            )

        return output

    except Exception as e:
        return f"ERROR: {e}"
