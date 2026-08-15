"""Errors raised by the controller and the agent layer."""

from __future__ import annotations

from nexus_errors.base_error import NexusError


class ControllerError(NexusError):
    """Base for controller failures."""

    code = "controller_error"


class PolicyLoadError(ControllerError):
    """A checkpoint could not be loaded.

    Most often an ``MDP_VERSION`` mismatch: a policy trained under one observation/action/reward contract is
    meaningless under another, so loading it is refused rather than silently producing nonsense.
    """

    code = "policy_load_error"


class ActionDecodingError(ControllerError):
    """A policy output could not be decoded into a valid allocation.

    Usually a shape mismatch between the policy's action space and the current environment. Symptom to watch
    for: all-zero allocations that look like a flat reward curve.
    """

    code = "action_decoding_error"


class AgentError(NexusError):
    """Base for agent-layer failures."""

    code = "agent_error"


class ConstraintRejectedError(AgentError):
    """The controller refused a constraint.

    Raised when a TTL exceeds ``MAX_CONSTRAINT_TTL``, a weight is out of range, or a mask would forbid every
    action. Surfaces to the model as a tool error so it reports the refusal instead of assuming success.
    """

    code = "constraint_rejected"


class ToolExecutionError(AgentError):
    """An MCP tool failed.

    Always raise this rather than returning an empty-success payload. Given nothing, the model produces a
    fluent, confident, entirely invented answer indistinguishable from a real one.
    """

    code = "tool_execution_error"
