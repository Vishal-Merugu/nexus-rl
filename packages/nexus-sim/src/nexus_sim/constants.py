"""Simulation constants.

Every magic number in the simulator lives here. Changing one changes the dynamics the policy trained
against, which usually invalidates existing checkpoints even when the MDP signature is untouched.
"""

from __future__ import annotations

TIMESTEP_SECONDS: float = 0.1
"""Wall-clock duration one timestep represents. Used to convert Mb/s into Mb of queue movement.

The simulator itself has no notion of real time — this is purely the unit conversion that makes queue
backlogs interpretable in megabits.
"""

BASE_PROPAGATION_MS: float = 1.0
"""Per-link propagation delay on an idle link."""

MAX_UTILIZATION_FOR_DELAY: float = 0.99
"""Utilization ceiling used in the queueing-delay term.

The 1/(1-u) factor diverges at u = 1. Clamping bounds the delay instead of producing an infinity that
propagates into the reward as a NaN and silently destroys a training run.
"""

MAX_LINK_LATENCY_MS: float = 500.0
"""Upper bound on any single link's derived latency. Keeps reward terms in a comparable range."""

MAX_QUEUE_MB: float = 50.0
"""Per-link queue capacity. Backlog beyond this is dropped.

A finite queue is what makes drops a real consequence. With an unbounded queue the policy learns to defer
everything forever and never pays for it.
"""

QUEUE_DRAIN_PRIORITY: float = 0.5
"""Fraction of a link's capacity reserved for draining existing backlog before new traffic is served.

Without this a congested link starves its own queue: new arrivals consume all capacity and the backlog never
clears, which looks like a policy failure but is a simulation artifact.
"""

EPSILON: float = 1e-9
"""Guard against division by zero on empty or failed links."""

DEFAULT_FLOW_COUNT: int = 8
DEFAULT_CANDIDATE_PATHS: int = 3
"""Paths offered per flow. The discrete action space grows with this — keep it small."""

DIURNAL_PERIOD_TIMESTEPS: int = 480
BURST_PROBABILITY: float = 0.06
BURST_MULTIPLIER: float = 3.5
FLASH_CROWD_START: int = 200
FLASH_CROWD_MULTIPLIER: float = 6.0
DEMAND_NOISE_STD: float = 0.08
