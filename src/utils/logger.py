#!/usr/bin/env python3
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

# ANSI colors (no external deps). Disabled for file logs and non-TTY streams.
RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"

# Optional custom level for SUCCESS (between INFO and WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class ColorFormatter(logging.Formatter):
    """
    Colorizes level name + message for console.
    Leaves file logs uncolored by using a standard Formatter there.
    """
    LEVEL_TO_COLOR = {
        logging.ERROR: RED,
        logging.WARNING: YELLOW,
        SUCCESS_LEVEL: GREEN,
        logging.INFO: "",   # normal
        logging.DEBUG: "",  # normal
    }

    def __init__(self, fmt: str, datefmt: Optional[str] = None, use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if not self.use_color:
            return msg
        color = self.LEVEL_TO_COLOR.get(record.levelno, "")
        if color:
            # Colorize only the message part; timestamp/module stay plain.
            # Assumes format starts with time/level prefix.
            return f"{color}{msg}{RESET}"
        return msg


class ModuleLogger:
    """
    Create a module-scoped logger with:
      - console handler (colored)
      - file handler (plain)
      - file log overwritten each run
    Usage:
        logger = ModuleLogger.get(__name__, log_dir='logs', filename='dataset_profiler.log').logger
        logger.info("Hello")
        logger.success("Step OK")  # custom
    """
    def __init__(
        self,
        name: str,
        log_dir: str | Path = "logs",
        filename: str = "dataset_profiler.log",
        level: int = logging.INFO,
        overwrite: bool = True,
        propagate: bool = False,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = propagate

        # Prevent duplicate handlers if called multiple times
        if self.logger.handlers:
            return

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / filename

        # --- Console handler (colored)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        use_color = sys.stdout.isatty()
        console.setFormatter(
            ColorFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                use_color=use_color,
            )
        )

        # --- File handler (plain, overwritten each run)
        file_mode = "w" if overwrite else "a"
        file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        self.logger.addHandler(console)
        self.logger.addHandler(file_handler)

        # Add 'success' convenience method and level
        def success(msg: str, *args, **kwargs):
            if self.logger.isEnabledFor(SUCCESS_LEVEL):
                self.logger.log(SUCCESS_LEVEL, msg, *args, **kwargs)

        if not hasattr(self.logger, "success"):
            setattr(self.logger, "success", success)  # type: ignore[attr-defined]

    @classmethod
    def get(
        cls,
        name: str,
        log_dir: str | Path = "logs",
        filename: str = "dataset_profiler.log",
        level: int = logging.INFO,
        overwrite: bool = True,
        propagate: bool = False,
    ) -> "ModuleLogger":
        return cls(
            name=name,
            log_dir=log_dir,
            filename=filename,
            level=level,
            overwrite=overwrite,
            propagate=propagate,
        )
