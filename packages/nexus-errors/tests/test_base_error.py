from __future__ import annotations

import pytest

from nexus_errors.base_error import NexusError, error_to_string
from nexus_errors.config_error import MdpVersionMismatchError
from nexus_errors.simulation_error import InvariantViolationError


def test_message_alone_renders_without_context_suffix() -> None:
    assert str(NexusError("link overloaded")) == "link overloaded"


def test_context_renders_sorted_and_appended() -> None:
    error = NexusError("link overloaded", {"utilization": 1.4, "link_id": "a->b"})

    assert str(error) == "link overloaded (link_id='a->b', utilization=1.4)"


def test_to_dict_carries_code_for_boundary_mapping() -> None:
    error = InvariantViolationError("queue went negative", {"queue_mb": -3.0})

    assert error.to_dict() == {
        "code": "invariant_violation",
        "message": "queue went negative",
        "context": {"queue_mb": -3.0},
    }


def test_subclasses_stay_catchable_as_nexus_error() -> None:
    with pytest.raises(NexusError):
        raise InvariantViolationError("utilization above one")


def test_mdp_mismatch_names_both_versions_and_the_artifact() -> None:
    error = MdpVersionMismatchError(expected=4, found=2, artifact="runs/a1b2c3/policy")

    assert error.context == {"expected": 4, "found": 2, "artifact": "runs/a1b2c3/policy"}
    assert "MDP_VERSION 2" in error.message
    assert "version 4" in error.message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (NexusError("boom"), "boom"),
        (ValueError("bad input"), "ValueError: bad input"),
        ("not an exception", "'not an exception'"),
    ],
)
def test_error_to_string_handles_unknown_shapes(value: object, expected: str) -> None:
    assert error_to_string(value) == expected
