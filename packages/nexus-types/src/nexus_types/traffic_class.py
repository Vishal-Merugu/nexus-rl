"""Traffic classes — the QoS category of a flow."""

from __future__ import annotations

from enum import StrEnum


class TrafficClass(StrEnum):
    """The QoS category of a flow.

    Determines the weight a flow's latency and throughput carry in the reward, and which flows an operator
    constraint can target. This is an *input* property of the flow — not the priority the controller chooses,
    which is an outcome.
    """

    LATENCY_SENSITIVE = "LATENCY_SENSITIVE"
    """Interactive traffic. Latency dominates its contribution to the reward."""

    BULK = "BULK"
    """Throughput-oriented transfers. Tolerates delay, penalised for starvation."""

    BACKGROUND = "BACKGROUND"
    """Best-effort. Yields to everything else; only drops are penalised."""
