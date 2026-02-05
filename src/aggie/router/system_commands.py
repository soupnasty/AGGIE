"""System commands that don't require an LLM."""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemResponse:
    """Response from a system command."""

    text: str
    command: str  # Which command matched


# Patterns for system commands (compiled for efficiency)
TIME_PATTERNS = [
    re.compile(r"\bwhat time\b", re.IGNORECASE),
    re.compile(r"\bcurrent time\b", re.IGNORECASE),
    re.compile(r"\bwhat's the time\b", re.IGNORECASE),
    re.compile(r"\btell me the time\b", re.IGNORECASE),
]

DATE_PATTERNS = [
    re.compile(r"\bwhat's the date\b", re.IGNORECASE),
    re.compile(r"\bwhat date\b", re.IGNORECASE),
    re.compile(r"\btoday's date\b", re.IGNORECASE),
    re.compile(r"\bwhat day is it\b", re.IGNORECASE),
    re.compile(r"\bwhat day is today\b", re.IGNORECASE),
]


def check_system_command(query: str) -> Optional[SystemResponse]:
    """Check if query matches a system command.

    Args:
        query: User query text.

    Returns:
        SystemResponse if matched, None otherwise.
    """
    # Check time patterns
    for pattern in TIME_PATTERNS:
        if pattern.search(query):
            time_str = datetime.now().strftime("%I:%M %p").lstrip("0")
            logger.info(f"System command matched: time -> {time_str}")
            return SystemResponse(
                text=f"It's {time_str}.",
                command="time",
            )

    # Check date patterns
    for pattern in DATE_PATTERNS:
        if pattern.search(query):
            date_str = datetime.now().strftime("%A, %B %d, %Y")
            logger.info(f"System command matched: date -> {date_str}")
            return SystemResponse(
                text=f"It's {date_str}.",
                command="date",
            )

    return None
