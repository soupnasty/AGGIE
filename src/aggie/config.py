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
    silence_threshold: float = 500.0
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
class SessionConfig:
    """Session context configuration."""

    soft_decay_minutes: int = 10
    hard_decay_minutes: int = 30
    max_session_tokens: int = 8000
    summarizer_model: str = "facebook/bart-large-cnn"


@dataclass
class RouterConfig:
    """Query router configuration."""

    enabled: bool = True  # Enable routing (False = always use Claude)
    llama_model: str = "llama3.1:8b-instruct-q4_0"  # Ollama model name
    llama_host: str = "http://localhost:11434"  # Ollama API URL
    claude_fallback: bool = True  # Fall back to Claude if Llama unavailable


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
    ipc: IPCConfig = field(default_factory=IPCConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

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

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            wakeword=WakeWordConfig(**data.get("wakeword", {})),
            stt=STTConfig(**data.get("stt", {})),
            llm=LLMConfig(**data.get("llm", {})),
            tts=TTSConfig(**data.get("tts", {})),
            ipc=IPCConfig(**data.get("ipc", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            session=SessionConfig(**data.get("session", {})),
            router=RouterConfig(**data.get("router", {})),
        )
