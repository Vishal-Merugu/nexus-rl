"""Structured logging. Never ``print()`` in library or service code.

Records are emitted as single-line JSON so a run's logs stay greppable and joinable on ``timestep``.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Protocol

from nexus_errors.base_error import NexusError, error_to_string


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload["context"] = extra
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TelemetryClient(Protocol):
    """The logging surface used across the repo.

    A protocol rather than a base class so tests can substitute a recorder without inheriting behaviour.
    """

    def info(self, message: str, context: dict[str, object] | None = None) -> None: ...

    def warning(self, message: str, context: dict[str, object] | None = None) -> None: ...

    def capture_exception(self, error: object, message: str) -> None: ...


class LoggingTelemetryClient:
    """Default client. Writes JSON lines to stderr."""

    def __init__(self, name: str, level: str = "INFO") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(handler)
        self._logger.setLevel(level)
        self._logger.propagate = False

    def info(self, message: str, context: dict[str, object] | None = None) -> None:
        self._logger.info(message, extra={"context": context or {}})

    def warning(self, message: str, context: dict[str, object] | None = None) -> None:
        self._logger.warning(message, extra={"context": context or {}})

    def capture_exception(self, error: object, message: str) -> None:
        """Log an exception with its structured context.

        Every ``except`` block either re-raises or calls this. A swallowed exception in a control loop
        produces a system that looks healthy while doing nothing.
        """
        context: dict[str, object] = {"error": error_to_string(error)}
        if isinstance(error, NexusError):
            context["code"] = error.code
            context["error_context"] = error.context
        self._logger.error(message, extra={"context": context})


def get_telemetry(name: str, level: str = "INFO") -> LoggingTelemetryClient:
    """Construct a client for a module or service."""
    return LoggingTelemetryClient(name, level)
