"""Security guards and logging helpers for AirMouse."""

from __future__ import annotations

import http.client
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import socket
import sys
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent
_LOGGER_NAME = "airmouse"
_NETWORK_GUARDS_INSTALLED = False


class NetworkAccessBlockedError(RuntimeError):
    """Raised when any code attempts to open a network connection."""


def ensure_within_project(path: Path) -> Path:
    """Resolve a path and ensure it stays inside the project directory."""

    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path escapes the project directory: {resolved}")
    return resolved


def install_network_guards() -> None:
    """Disable all socket and urllib access for the current process."""

    global _NETWORK_GUARDS_INSTALLED
    if _NETWORK_GUARDS_INSTALLED:
        return

    def _blocked(*_args, **_kwargs):
        raise NetworkAccessBlockedError(
            "Network access is disabled for AirMouse. "
            "Socket and urllib calls are not permitted."
        )

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    if hasattr(socket, "socketpair"):
        socket.socketpair = _blocked  # type: ignore[assignment]
    if hasattr(socket, "create_server"):
        socket.create_server = _blocked  # type: ignore[assignment]
    urllib.request.urlopen = _blocked  # type: ignore[assignment]
    urllib.request.build_opener = _blocked  # type: ignore[assignment]
    http.client.HTTPConnection.connect = _blocked  # type: ignore[assignment]
    http.client.HTTPSConnection.connect = _blocked  # type: ignore[assignment]

    _NETWORK_GUARDS_INSTALLED = True


def prepare_runtime_environment(log_dir_name: str) -> Path:
    """Constrain third-party runtime artifacts to the project-local logs directory."""

    log_dir = ensure_within_project(PROJECT_ROOT / log_dir_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    matplotlib_dir = ensure_within_project(log_dir / "matplotlib")
    matplotlib_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)
    os.environ["XDG_CACHE_HOME"] = str(log_dir)

    return log_dir


def configure_logging(
    *,
    production_mode: bool,
    enable_file_logging: bool,
    log_dir_name: str,
    log_file_name: str,
    log_file_max_bytes: int,
    log_backup_count: int,
    log_level: str,
) -> logging.Logger:
    """Configure the process-wide AirMouse logger once."""

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if enable_file_logging:
        log_dir = ensure_within_project(PROJECT_ROOT / log_dir_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = ensure_within_project(log_dir / log_file_name)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_file_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if production_mode:
        logging.getLogger("pyautogui").setLevel(logging.WARNING)

    return logger


def enforce_directory_file_size_cap(directory: Path, max_bytes: int) -> None:
    """Remove any file under a project-local directory that exceeds the size cap."""

    target_directory = ensure_within_project(directory)
    if not target_directory.exists():
        return

    for file_path in target_directory.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.stat().st_size <= max_bytes:
            continue
        file_path.unlink(missing_ok=True)
