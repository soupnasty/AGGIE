"""Entry point for the AGGIE daemon."""

import asyncio
import argparse
import logging
import sys

from .config import Config
from .daemon import AggieDaemon


def setup_logging(config: Config) -> None:
    """Configure logging based on config."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.logging.level.upper()))

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(console)

    if config.logging.file:
        file_handler = logging.FileHandler(config.logging.file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="aggie",
        description="AGGIE - Privacy-first voice assistant daemon",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.load(args.config)

    if args.debug:
        config.logging.level = "DEBUG"

    setup_logging(config)

    # Run the daemon
    daemon = AggieDaemon(config)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
