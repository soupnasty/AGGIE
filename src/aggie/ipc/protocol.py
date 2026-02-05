"""IPC protocol definitions for daemon-CLI communication."""

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Union


class CommandType(str, Enum):
    """Available IPC commands."""

    MUTE = "mute"
    UNMUTE = "unmute"
    CANCEL = "cancel"
    STATUS = "status"
    SHUTDOWN = "shutdown"
    DEBUG_DUMP = "debug_dump"


class ResponseStatus(str, Enum):
    """Response status codes."""

    OK = "ok"
    ERROR = "error"


@dataclass
class Command:
    """Command sent from CLI to daemon."""

    type: CommandType

    def to_json(self) -> str:
        """Serialize command to JSON."""
        return json.dumps({"type": self.type.value})

    @classmethod
    def from_json(cls, data: str) -> "Command":
        """Deserialize command from JSON."""
        parsed = json.loads(data)
        return cls(type=CommandType(parsed["type"]))


@dataclass
class StatusResponse:
    """Status information from daemon."""

    status: ResponseStatus
    state: str
    muted: bool
    uptime_seconds: float
    gpu: Optional[str] = None  # GPU info if available (e.g., "GTX 1060 (6.0GB)")
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize response to JSON."""
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "StatusResponse":
        """Deserialize response from JSON."""
        parsed = json.loads(data)
        parsed["status"] = ResponseStatus(parsed["status"])
        return cls(**parsed)


@dataclass
class SimpleResponse:
    """Simple OK/Error response."""

    status: ResponseStatus
    message: str
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize response to JSON."""
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "SimpleResponse":
        """Deserialize response from JSON."""
        parsed = json.loads(data)
        parsed["status"] = ResponseStatus(parsed["status"])
        return cls(**parsed)


@dataclass
class DebugDumpResponse:
    """Debug log dump response."""

    status: ResponseStatus
    log_path: str
    lines: int
    content: str
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize response to JSON."""
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> "DebugDumpResponse":
        """Deserialize response from JSON."""
        parsed = json.loads(data)
        parsed["status"] = ResponseStatus(parsed["status"])
        return cls(**parsed)


# Type alias for any response
Response = Union[StatusResponse, SimpleResponse, DebugDumpResponse]
