"""Bounds shared across service boundaries.

Magic numbers belong here or in a config file, never inline at a call site.
"""

from __future__ import annotations

MAX_CONSTRAINT_TTL: int = 3_000
"""Upper bound on a constraint's lifetime, in timesteps.

Every operator override expires. An unbounded one silently becomes permanent policy that nobody remembers
setting — and, in a system whose selling point is autonomy, that is the failure that discredits it.
"""

MAX_OBJECTIVE_WEIGHT: float = 10.0
"""Upper bound on any single objective weight.

Keeps reward terms within comparable magnitudes. A term two orders larger than the rest makes the others
invisible to the policy, which then optimizes the only thing it can see.
"""

DEFAULT_HISTORY_SIZE: int = 512
"""Timesteps of state the controller retains for agent explanations.

Bounded on purpose: the ring buffer is a debugging and explanation aid, not a datastore.
"""

DEFAULT_EPISODE_LENGTH: int = 512
"""Timesteps per training episode."""

DASHBOARD_DOWNSAMPLE: int = 5
"""Emit every Nth timestep to the dashboard.

The simulator steps far faster than a browser can paint. Streaming raw training-speed state to a WebSocket
gains nothing and stalls the client.
"""
