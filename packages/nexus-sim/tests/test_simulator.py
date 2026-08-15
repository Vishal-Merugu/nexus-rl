from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexus_sim.simulator import NetworkSimulator
from nexus_types.action import AllocationAction
from nexus_types.incident import Incident, IncidentKind


def _simulator(pattern: str = "steady", incidents: tuple[Incident, ...] = ()) -> NetworkSimulator:
    return NetworkSimulator.from_config(
        topology_name="mesh-6", traffic_pattern_name=pattern, seed=1, incidents=incidents
    )


def test_simulator_starts_at_timestep_zero_with_a_full_state() -> None:
    simulator = _simulator()

    assert simulator.state.timestep == 0
    assert len(simulator.state.links) == len(simulator.topology.links)
    assert len(simulator.state.flows) == len(simulator.flows)


def test_step_advances_the_timestep() -> None:
    simulator = _simulator()

    assert simulator.step().state.timestep == 1
    assert simulator.step().state.timestep == 2


def test_emitted_state_is_frozen_so_snapshots_stay_valid() -> None:
    state = _simulator().step().state

    with pytest.raises(ValidationError):
        state.timestep = 99  # type: ignore[misc]  # the point of the test is that this is illegal


def test_a_previously_handed_out_snapshot_is_not_mutated_by_a_later_step() -> None:
    simulator = _simulator()
    snapshot = simulator.step().state
    captured_timestep = snapshot.timestep
    captured_throughput = snapshot.total_allocated_mbps

    for _ in range(5):
        simulator.step()

    assert snapshot.timestep == captured_timestep
    assert snapshot.total_allocated_mbps == captured_throughput


def test_invariants_hold_over_a_long_run() -> None:
    simulator = _simulator(pattern="bursty")

    for _ in range(400):
        state = simulator.step().state
        for link in state.links:
            assert 0.0 <= link.utilization <= 1.0
            assert link.queue_mb >= 0.0
            assert link.allocated_mbps <= link.effective_capacity_mbps + 1e-3
        for flow in state.flows:
            assert flow.allocated_mbps <= flow.demand_mbps + 1e-3
            assert flow.dropped_mbps >= 0.0


def test_reset_clears_accumulated_backlog() -> None:
    # reset() zeroes the queues and then simulates timestep 0, so a congested topology may immediately
    # re-accumulate a little. The contract is that backlog does not carry across a reset — not that the
    # emitted state has literally empty queues.
    simulator = _simulator(pattern="flash_crowd")
    for _ in range(250):
        simulator.step()
    congested_backlog = sum(link.queue_mb for link in simulator.state.links)

    state = simulator.reset()

    assert state.timestep == 0
    assert sum(link.queue_mb for link in state.links) < congested_backlog


def test_throttling_every_flow_reduces_throughput() -> None:
    open_sim = _simulator()
    throttled_sim = _simulator()
    throttle = AllocationAction(share_by_flow_id={flow.flow_id: 0.1 for flow in throttled_sim.flows})

    open_total = sum(open_sim.step().metrics.total_throughput_mbps for _ in range(20))
    throttled_total = sum(throttled_sim.step(throttle).metrics.total_throughput_mbps for _ in range(20))

    assert throttled_total < open_total


def test_an_action_naming_unknown_flows_still_produces_a_valid_state() -> None:
    # Action decoding must be total — a partial or stale action is a bug worth finding, not one worth
    # crashing a training run over.
    simulator = _simulator()

    result = simulator.step(AllocationAction(share_by_flow_id={"ghost-flow": 0.5}))

    assert result.state.timestep == 1


def test_a_failed_link_removes_its_capacity() -> None:
    healthy = _simulator()
    target_link = healthy.topology.links[0].link_id
    degraded = _simulator(
        incidents=(
            Incident(
                incident_id="inc-1",
                kind=IncidentKind.LINK_FAILED,
                target_id=target_link,
                start_timestep=1,
                duration_timesteps=10,
            ),
        )
    )

    state = degraded.step().state

    assert state.link_by_id(target_link).effective_capacity_mbps == 0.0
    assert "inc-1" in state.active_incident_ids


def test_capacity_returns_when_the_incident_expires() -> None:
    target_link = _simulator().topology.links[0].link_id
    simulator = _simulator(
        incidents=(
            Incident(
                incident_id="inc-1",
                kind=IncidentKind.LINK_FAILED,
                target_id=target_link,
                start_timestep=1,
                duration_timesteps=2,
            ),
        )
    )

    simulator.step()  # t=1, degraded
    simulator.step()  # t=2, degraded
    state = simulator.step()  # t=3, recovered

    link = state.state.link_by_id(target_link)
    assert link.effective_capacity_mbps == link.capacity_mbps
    assert state.state.active_incident_ids == ()


def test_a_node_failure_takes_down_every_incident_link() -> None:
    simulator = _simulator(
        incidents=(
            Incident(
                incident_id="node-down",
                kind=IncidentKind.NODE_FAILED,
                target_id="n0",
                start_timestep=1,
                duration_timesteps=5,
            ),
        )
    )

    state = simulator.step().state

    for link in state.links:
        if "n0" in (link.source, link.target):
            assert link.effective_capacity_mbps == 0.0


def test_a_degraded_link_keeps_a_fraction_of_capacity() -> None:
    target_link = _simulator().topology.links[0].link_id
    simulator = _simulator(
        incidents=(
            Incident(
                incident_id="inc-slow",
                kind=IncidentKind.LINK_DEGRADED,
                target_id=target_link,
                start_timestep=1,
                duration_timesteps=5,
                capacity_factor=0.25,
            ),
        )
    )

    link = simulator.step().state.link_by_id(target_link)

    assert link.effective_capacity_mbps == pytest.approx(link.capacity_mbps * 0.25)


def test_link_carried_rate_reconciles_with_flow_throughput_plus_drain() -> None:
    # Regression: per-link carried rate exceeded per-flow throughput with no explanation, because backlog
    # drain is not attributed to any flow. The gap must now be reported, not silent.
    simulator = _simulator(pattern="flash_crowd")
    for _ in range(250):
        simulator.step()

    state, metrics = simulator.state, simulator.metrics
    link_carried = sum(link.allocated_mbps for link in state.links)
    link_drained = sum(link.drained_mbps for link in state.links)

    assert metrics.total_drained_mbps == pytest.approx(link_drained, rel=1e-4)
    assert metrics.total_delivered_mbps == pytest.approx(
        metrics.total_throughput_mbps + metrics.total_drained_mbps, rel=1e-6
    )
    assert link_carried >= metrics.total_throughput_mbps - 1e-3


def test_metrics_track_the_state_they_were_computed_from() -> None:
    simulator = _simulator()

    result = simulator.step()

    assert result.metrics.timestep == result.state.timestep
    assert result.metrics.total_throughput_mbps == pytest.approx(result.state.total_allocated_mbps, rel=1e-4)


def test_fairness_stays_within_bounds() -> None:
    simulator = _simulator(pattern="flash_crowd")

    for _ in range(300):
        assert 0.0 < simulator.step().metrics.fairness_index <= 1.0


def test_flows_report_the_path_they_actually_took() -> None:
    simulator = _simulator()
    flow = simulator.flows[0]
    alternate = flow.candidate_paths[-1]

    state = simulator.step(
        AllocationAction(share_by_flow_id={}, path_by_flow_id={flow.flow_id: alternate})
    ).state

    assert next(f for f in state.flows if f.flow_id == flow.flow_id).path == alternate


def test_an_incident_that_matches_nothing_is_not_reported_as_active() -> None:
    # A ghost incident used to appear in active_incident_ids while degrading nothing, so the dashboard
    # showed "Incident active" and the agent would tell an operator the network was degraded.
    simulator = _simulator(
        incidents=(
            Incident(
                incident_id="ghost",
                kind=IncidentKind.LINK_FAILED,
                target_id="nowhere->nohow",
                start_timestep=0,
                duration_timesteps=50,
            ),
        )
    )

    state = simulator.step().state

    assert state.active_incident_ids == ()
    assert all(link.effective_capacity_mbps == link.capacity_mbps for link in state.links)


def test_a_real_incident_is_still_reported() -> None:
    target = _simulator().topology.links[0].link_id
    simulator = _simulator(
        incidents=(
            Incident(
                incident_id="real",
                kind=IncidentKind.LINK_FAILED,
                target_id=target,
                start_timestep=0,
                duration_timesteps=50,
            ),
        )
    )

    assert simulator.step().state.active_incident_ids == ("real",)
