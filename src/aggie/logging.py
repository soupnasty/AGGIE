"""Two-tier structured logging for AGGIE.

Provides:
- Console: Minimal output (INFO+) for human readability
- Debug file: JSON Lines format with full context for LLM debugging
"""

import json
import logging
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .state import State

# Global reference to current state for context injection
_current_state: State = State.IDLE


def set_current_state(state: State) -> None:
    """Update the current state for log context injection."""
    global _current_state
    _current_state = state


def get_current_state() -> State:
    """Get the current state for log context."""
    return _current_state


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON Lines for structured debugging.

    Output format:
    {"ts":"2026-02-04T10:15:32.123","level":"DEBUG","component":"wakeword",
     "state":"IDLE","msg":"confidence=0.87","ctx":{...}}
    """

    def format(self, record: logging.LogRecord) -> str:
        # Extract component name from logger name (e.g., "aggie.detection.wakeword" -> "wakeword")
        component = record.name.split(".")[-1] if "." in record.name else record.name

        log_entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": component,
            "state": _current_state.name,
            "msg": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exc"] = self.formatException(record.exc_info)

        # Add extra context if provided via extra={"ctx": {...}}
        if hasattr(record, "ctx"):
            log_entry["ctx"] = record.ctx

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Clean, minimal console formatter for human readability."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Short level names
        level_short = {
            "DEBUG": "DBG",
            "INFO": "INF",
            "WARNING": "WRN",
            "ERROR": "ERR",
            "CRITICAL": "CRT",
        }.get(record.levelname, record.levelname[:3])

        # Extract component
        component = record.name.split(".")[-1] if "." in record.name else record.name

        # Format timestamp as HH:MM:SS
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Build message
        msg = f"{time_str} [{level_short}] {component}: {record.getMessage()}"

        # Add colors if enabled
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            msg = f"{color}{msg}{self.RESET}"

        # Add exception if present
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def get_debug_log_path() -> Path:
    """Get the path to the debug log file (XDG compliant)."""
    data_home = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    log_dir = Path(data_home) / "aggie" / "logs"
    return log_dir / "debug.log"


def rotate_debug_log(log_path: Path) -> None:
    """Rotate existing debug log to .1 on startup."""
    if log_path.exists():
        rotated = log_path.with_suffix(".log.1")
        # Remove old rotated log
        if rotated.exists():
            rotated.unlink()
        # Rotate current to .1
        log_path.rename(rotated)


def log_session_header(config: Any, logger: logging.Logger) -> None:
    """Log session startup info for self-contained debug logs."""
    header = {
        "session_start": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_version": platform.release(),
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else str(config),
    }
    logger.info(
        "=== AGGIE Session Started ===",
        extra={"ctx": header}
    )


def setup_logging(
    config: Any,
    console_level: str = "INFO",
    debug_to_file: bool = True,
    use_colors: bool = True,
) -> None:
    """Configure two-tier logging system.

    Args:
        config: AGGIE Config object for session header
        console_level: Minimum level for console output (default: INFO)
        debug_to_file: Whether to write JSON debug logs to file
        use_colors: Whether to use ANSI colors in console output
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture everything, filter at handler level

    # Clear any existing handlers
    root.handlers.clear()

    # === Console Handler (minimal, human-readable) ===
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, console_level.upper()))
    console.setFormatter(ConsoleFormatter(use_colors=use_colors))
    root.addHandler(console)

    # === Debug File Handler (JSON Lines, full detail) ===
    if debug_to_file:
        log_path = get_debug_log_path()

        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotate existing log
        rotate_debug_log(log_path)

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # === Silence noisy third-party libraries ===
    for lib in ["httpx", "anthropic", "faster_whisper", "urllib3", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Log session header with config
    aggie_logger = logging.getLogger("aggie")
    log_session_header(config, aggie_logger)


def get_debug_log_contents(lines: int = 200) -> str:
    """Read the last N lines of the debug log for Claude debugging.

    Args:
        lines: Number of lines to return (default: 200)

    Returns:
        String containing the last N lines of the debug log
    """
    log_path = get_debug_log_path()
    if not log_path.exists():
        return "No debug log found."

    with open(log_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    return "".join(all_lines[-lines:])


def get_filtered_logs(
    component: Optional[str] = None,
    level: Optional[str] = None,
    lines: int = 100,
) -> str:
    """Get filtered debug logs for specific debugging needs.

    Args:
        component: Filter to specific component (e.g., "wakeword", "stt")
        level: Filter to specific level (e.g., "ERROR", "WARNING")
        lines: Maximum lines to return

    Returns:
        Filtered log lines as string
    """
    log_path = get_debug_log_path()
    if not log_path.exists():
        return "No debug log found."

    results = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if component and entry.get("component") != component:
                    continue
                if level and entry.get("level") != level:
                    continue
                results.append(line)
            except json.JSONDecodeError:
                continue

    return "".join(results[-lines:])
