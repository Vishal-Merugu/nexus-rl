from __future__ import annotations

import pytest

from nexus_errors.control_error import ActionDecodingError
from nexus_types.action import AllocationAction


def test_shares_within_bounds_are_accepted() -> None:
    action = AllocationAction(share_by_flow_id={"f1": 0.0, "f2": 1.0, "f3": 0.42})

    assert action.share_by_flow_id["f3"] == 0.42


@pytest.mark.parametrize("bad_share", [-0.01, 1.01, 12.0])
def test_out_of_range_share_is_rejected_as_a_decoding_error(bad_share: float) -> None:
    with pytest.raises(ActionDecodingError):
        AllocationAction(share_by_flow_id={"f1": bad_share})


def test_provenance_defaults_to_the_controller() -> None:
    assert AllocationAction(share_by_flow_id={}).issued_by == "controller"


def test_baseline_fallback_records_itself_in_provenance() -> None:
    action = AllocationAction(share_by_flow_id={"f1": 0.5}, issued_by="baseline:static")

    assert action.issued_by == "baseline:static"
