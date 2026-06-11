"""Configuración centralizada del logging del proyecto."""
from __future__ import annotations

import logging
import sys
import time


class _ColorFormatter(logging.Formatter):
    """Formatter con colores ANSI por nivel (se desactiva si no hay TTY)."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # cian
        logging.INFO: "\033[32m",      # verde
        logging.WARNING: "\033[33m",   # amarillo
        logging.ERROR: "\033[31m",     # rojo
        logging.CRITICAL: "\033[41m",  # fondo rojo
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool):
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"
        return msg


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configura el logger raíz del proyecto.

    verbose -> nivel DEBUG, quiet -> WARNING, por defecto INFO.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger("tennis")
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_ColorFormatter(use_color=sys.stdout.isatty()))
    root.addHandler(handler)
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Logger hijo del namespace 'tennis' (p.ej. get_logger(__name__))."""
    short = name.split(".")[-1]
    return logging.getLogger(f"tennis.{short}")


class StepTimer:
    """Context manager que loguea inicio/fin y duración de una fase."""

    def __init__(self, logger: logging.Logger, msg: str):
        self.logger = logger
        self.msg = msg

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.info("▶ %s ...", self.msg)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            self.logger.info("✔ %s (%.2fs)", self.msg, dt)
        else:
            self.logger.error("✘ %s falló tras %.2fs: %s", self.msg, dt, exc)
        return False
