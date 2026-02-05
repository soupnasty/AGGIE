"""Entry point for the AGGIE daemon."""

import asyncio
import argparse

from .config import Config
from .daemon import AggieDaemon
from .logging import setup_logging


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

    setup_logging(
        config,
        console_level=config.logging.level,
        debug_to_file=config.logging.debug_to_file,
        use_colors=config.logging.use_colors,
    )

    # Run the daemon
    daemon = AggieDaemon(config)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
