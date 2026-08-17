"""Fallback activation, and the hysteresis that stops it oscillating.

The naive version of this — "revert when latency is high, return when it is not" — produces a system that
flaps between two allocators, which is worse than either alone. The fallback lowers latency; the moment it
does, the policy gets the network back and breaks it again.

So the tests here are mostly about *not* acting: not on one bad step, not the instant things recover.
"""

from __future__ import annotations

from nexus_safety.fallback import FallbackController, FallbackReason, QosSample, QosThresholds

_HEALTHY = QosSample(p95_latency_ms=10.0, drop_rate=0.01, fairness=0.9)
_BREACHING = QosSample(p95_latency_ms=900.0, drop_rate=0.01, fairness=0.9)

_FAST = QosThresholds(consecutive_breaches_to_engage=3, consecutive_healthy_to_release=5)


def _run(controller: FallbackController, sample: QosSample, times: int, start: int = 0) -> None:
    for offset in range(times):
        controller.observe(sample, start + offset)


def test_a_healthy_network_leaves_the_policy_driving() -> None:
    controller = FallbackController(_FAST)

    _run(controller, _HEALTHY, 50)

    assert not controller.engaged
    assert controller.activation_count == 0


def test_one_bad_timestep_does_not_trigger_a_fallback() -> None:
    # A single burst is not a failing policy.
    controller = FallbackController(_FAST)

    controller.observe(_BREACHING, 1)

    assert not controller.engaged


def test_a_sustained_breach_engages_the_fallback() -> None:
    controller = FallbackController(_FAST)

    _run(controller, _BREACHING, 3)

    assert controller.engaged
    assert controller.reason is FallbackReason.QOS_BREACH
    assert controller.activation_count == 1


def test_an_intermittent_breach_never_accumulates() -> None:
    # Alternating breach/healthy must not reach the engage threshold — the counter resets, not decays.
    controller = FallbackController(_FAST)

    for step in range(40):
        controller.observe(_BREACHING if step % 2 == 0 else _HEALTHY, step)

    assert not controller.engaged


def test_recovery_does_not_hand_control_straight_back() -> None:
    # The whole hysteresis argument: the fallback fixed the metric, so of course it looks healthy now.
    controller = FallbackController(_FAST)
    _run(controller, _BREACHING, 3)

    _run(controller, _HEALTHY, 4, start=10)

    assert controller.engaged


def test_the_policy_regains_control_after_the_full_cooldown() -> None:
    controller = FallbackController(_FAST)
    _run(controller, _BREACHING, 3)

    _run(controller, _HEALTHY, 5, start=10)

    assert not controller.engaged
    assert controller.reason is FallbackReason.NONE


def test_a_breach_during_the_cooldown_restarts_it() -> None:
    controller = FallbackController(_FAST)
    _run(controller, _BREACHING, 3)
    _run(controller, _HEALTHY, 4, start=10)

    controller.observe(_BREACHING, 20)
    _run(controller, _HEALTHY, 4, start=21)

    assert controller.engaged


def test_an_unsafe_action_engages_immediately_without_a_window() -> None:
    # Waiting ten timesteps to react to a NaN would mean applying it ten times.
    controller = FallbackController(_FAST)

    controller.engage_now(FallbackReason.UNSAFE_ACTION, "non-finite share", 5)

    assert controller.engaged
    assert controller.reason is FallbackReason.UNSAFE_ACTION


def test_an_unreachable_controller_is_a_distinct_reason_from_a_bad_action() -> None:
    # Nothing is wrong with the policy and the fix is elsewhere; conflating them misdirects the operator.
    controller = FallbackController(_FAST)

    controller.engage_now(FallbackReason.CONTROLLER_UNREACHABLE, "no response", 5)

    assert controller.reason is FallbackReason.CONTROLLER_UNREACHABLE


def test_repeated_immediate_engagements_for_one_reason_count_once() -> None:
    # Otherwise the activation count — the system's clearest failure signal — inflates by one per timestep
    # and stops meaning anything.
    controller = FallbackController(_FAST)

    for step in range(10):
        controller.engage_now(FallbackReason.UNSAFE_ACTION, "non-finite share", step)

    assert controller.activation_count == 1


def test_a_change_of_reason_is_a_new_activation() -> None:
    controller = FallbackController(_FAST)
    controller.engage_now(FallbackReason.UNSAFE_ACTION, "nan", 1)

    controller.engage_now(FallbackReason.CONTROLLER_UNREACHABLE, "gone", 2)

    assert controller.activation_count == 2
    assert controller.reason is FallbackReason.CONTROLLER_UNREACHABLE


def test_every_transition_is_recorded_with_its_timestep() -> None:
    # The rate of these is the failure signal, so they are kept rather than merely acted on.
    controller = FallbackController(_FAST)
    _run(controller, _BREACHING, 3)
    _run(controller, _HEALTHY, 5, start=10)

    transitions = controller.transitions

    assert [t.engaged for t in transitions] == [True, False]
    assert transitions[0].timestep == 2


def test_drop_rate_and_fairness_are_breachable_independently_of_latency() -> None:
    # A policy serving one flow and starving the rest keeps latency low and is still failing.
    unfair = QosSample(p95_latency_ms=5.0, drop_rate=0.01, fairness=0.05)
    controller = FallbackController(_FAST)

    _run(controller, unfair, 3)

    assert controller.engaged

    dropping = QosSample(p95_latency_ms=5.0, drop_rate=0.9, fairness=0.9)
    other = FallbackController(_FAST)
    _run(other, dropping, 3)

    assert other.engaged


def test_status_reports_the_thresholds_it_is_judging_against() -> None:
    # An operator seeing "fallback engaged" needs to know what line was crossed without reading the code.
    status = FallbackController(_FAST).status()

    assert status["fallback_engaged"] is False
    assert isinstance(status["thresholds"], dict)
    assert status["thresholds"]["consecutive_breaches_to_engage"] == 3
