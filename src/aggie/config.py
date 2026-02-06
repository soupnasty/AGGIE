"""Configuration loading and validation for AGGIE."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

import yaml


@dataclass
class AudioConfig:
    """Audio capture and playback configuration."""

    sample_rate: int = 16000
    channels: int = 1
    pre_roll_seconds: float = 1.5
    silence_threshold: float = 2000.0
    silence_duration: float = 1.5
    max_recording_duration: float = 30.0
    # Audio feedback cues (chimes for wake word, done listening, etc.)
    cues_enabled: bool = False


@dataclass
class WakeWordConfig:
    """Wake word detection configuration."""

    model: str = "hey_aggie"
    threshold: float = 0.5
    vad_threshold: float = 0.5


@dataclass
class STTConfig:
    """Speech-to-text configuration."""

    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "en"


@dataclass
class LLMConfig:
    """LLM client configuration."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 300
    timeout: float = 30.0  # Request timeout in seconds
    max_retries: int = 3  # Max retries for transient errors


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""

    voice_model: str = "en_US-lessac-medium"
    speaking_rate: float = 1.0
    use_cuda: bool = False


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Tool access configuration."""

    enabled: bool = True
    working_dir: str = "~"
    timeout: float = 30.0
    max_output_chars: int = 16000
    max_agent_turns: int = 10
    max_tokens: int = 4096
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)


@dataclass
class ContextConfig:
    """Three-tier context management configuration."""

    compress_every_turns: int = 7
    recent_window: int = 6
    haiku_model: str = "claude-haiku-4-5-20251001"
    max_context_tokens: int = 8000


@dataclass
class IPCConfig:
    """IPC server configuration."""

    socket_path: Optional[str] = None

    def get_socket_path(self) -> str:
        """Get the socket path, using default if not specified."""
        if self.socket_path:
            return self.socket_path
        # Try XDG_RUNTIME_DIR first, fall back to /tmp
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if runtime_dir and os.path.isdir(runtime_dir) and os.access(runtime_dir, os.W_OK):
            return f"{runtime_dir}/aggie.sock"
        return "/tmp/aggie.sock"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: Optional[str] = None  # Deprecated: use debug_to_file instead
    debug_to_file: bool = True  # Write JSON debug logs to ~/.local/share/aggie/logs/
    use_colors: bool = True  # ANSI colors in console output


@dataclass
class Config:
    """Main configuration container."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    ipc: IPCConfig = field(default_factory=IPCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Load configuration from file.

        Args:
            path: Optional path to config file. If not provided, searches
                  XDG config locations.

        Returns:
            Loaded configuration with defaults for missing values.
        """
        config_path: Optional[Path] = None

        if path:
            config_path = Path(path)
        else:
            # Try XDG config first
            xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
            user_config = Path(xdg_config) / "aggie" / "config.yaml"

            if user_config.exists():
                config_path = user_config
            else:
                # Try system config
                system_config = Path("/etc/aggie/config.yaml")
                if system_config.exists():
                    config_path = system_config

        if config_path and config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            return cls._from_dict(data)

        return cls()

    @staticmethod
    def _parse_tools(data: dict) -> ToolConfig:
        """Parse tools config, handling nested mcp_servers list."""
        data = dict(data)  # shallow copy
        mcp_raw = data.pop("mcp_servers", [])
        config = ToolConfig(**data)
        config.mcp_servers = [MCPServerConfig(**s) for s in mcp_raw]
        return config

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            wakeword=WakeWordConfig(**data.get("wakeword", {})),
            stt=STTConfig(**data.get("stt", {})),
            llm=LLMConfig(**data.get("llm", {})),
            tts=TTSConfig(**data.get("tts", {})),
            tools=cls._parse_tools(data.get("tools", {})),
            ipc=IPCConfig(**data.get("ipc", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            context=ContextConfig(**data.get("context", {})),
        )
