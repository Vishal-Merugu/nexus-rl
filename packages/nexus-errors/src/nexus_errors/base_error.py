"""The root of the error hierarchy.

Every error raised anywhere in NEXUS-RL subclasses ``NexusError``. Never raise a bare ``Exception``,
``ValueError``, or ``RuntimeError`` — service boundaries map ``NexusError`` to structured responses, and
anything outside the hierarchy escapes as an unhandled 500 with no context attached.
"""

from __future__ import annotations


class NexusError(Exception):
    """Base class for every error in the system.

    Args:
        message: Human-readable description. Written for whoever reads the log at 2am, not for the compiler.
        context: Structured key/value detail attached to the error. Prefer this over string interpolation —
            it survives into logs as queryable fields.
    """

    #: Stable identifier used by service boundaries to choose a response code. Subclasses override it.
    code: str = "nexus_error"

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        return f"{self.message} ({detail})"

    def to_dict(self) -> dict[str, object]:
        """Serialize for a service response or a log record."""
        return {"code": self.code, "message": self.message, "context": self.context}


def error_to_string(error: object) -> str:
    """Extract a message from an exception of unknown shape.

    Use at boundaries where the caught object's type genuinely is not known — never as a substitute for
    catching the specific error you expect.
    """
    if isinstance(error, NexusError):
        return str(error)
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return repr(error)
