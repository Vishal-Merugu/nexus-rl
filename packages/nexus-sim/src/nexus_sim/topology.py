"""Network topology — nodes, directed links, and the paths between them.

A deliberately small custom graph model rather than a networkx dependency: the simulator's hot path runs
millions of times per training run, path enumeration happens once at construction, and everything else is a
numpy indexing problem. See ADR-0001.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nexus_errors.simulation_error import TopologyError


@dataclass(frozen=True, slots=True)
class Link:
    """A directed edge with fixed nominal capacity."""

    link_id: str
    source: str
    target: str
    capacity_mbps: float


@dataclass(frozen=True, slots=True)
class Topology:
    """An immutable directed graph with capacity on every edge."""

    name: str
    nodes: tuple[str, ...]
    links: tuple[Link, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise TopologyError("Topology has no nodes.", {"name": self.name})
        if not self.links:
            raise TopologyError("Topology has no links.", {"name": self.name})
        known = set(self.nodes)
        for link in self.links:
            if link.source not in known or link.target not in known:
                raise TopologyError(
                    "Link references a node not in the topology.",
                    {"link_id": link.link_id, "nodes": sorted(known)},
                )

    @property
    def link_ids(self) -> tuple[str, ...]:
        """Link ids in array order. This ordering is part of the observation contract — keep it stable."""
        return tuple(link.link_id for link in self.links)

    def link_index(self) -> dict[str, int]:
        """Map link_id → array index."""
        return {link.link_id: index for index, link in enumerate(self.links)}

    def capacities_mbps(self) -> NDArray[np.float32]:
        """Nominal capacities in array order."""
        return np.array([link.capacity_mbps for link in self.links], dtype=np.float32)

    def outgoing(self, node: str) -> tuple[Link, ...]:
        return tuple(link for link in self.links if link.source == node)

    def links_incident_to(self, node: str) -> tuple[str, ...]:
        """Every link touching a node, in either direction. Used to apply a NODE_FAILED incident."""
        return tuple(link.link_id for link in self.links if node in (link.source, link.target))

    def k_shortest_paths(self, source: str, target: str, k: int) -> tuple[tuple[str, ...], ...]:
        """Enumerate up to ``k`` simple paths from source to target, as ordered link ids.

        Breadth-first, so paths come out shortest-hop first. Ties are broken by link id, giving a stable
        ordering: candidate path 0 must mean the same route on every run, because the action space indexes
        into it.
        """
        if source == target:
            raise TopologyError("Path requested from a node to itself.", {"node": source})
        if k <= 0:
            raise TopologyError("k must be positive.", {"k": k})

        found: list[tuple[str, ...]] = []
        # queue entries: (current node, links taken, nodes visited)
        queue: deque[tuple[str, tuple[str, ...], frozenset[str]]] = deque([(source, (), frozenset({source}))])
        while queue and len(found) < k:
            node, taken, visited = queue.popleft()
            for link in sorted(self.outgoing(node), key=lambda item: item.link_id):
                if link.target in visited:
                    continue  # simple paths only — no loops
                path = (*taken, link.link_id)
                if link.target == target:
                    found.append(path)
                    if len(found) == k:
                        break
                    continue
                queue.append((link.target, path, visited | {link.target}))

        if not found:
            raise TopologyError(
                "No path between these nodes — the topology is disconnected for this flow.",
                {"source": source, "target": target, "topology": self.name},
            )
        return tuple(found)


def _mesh(name: str, node_count: int, capacity_mbps: float) -> Topology:
    """A ring with chords: every node links to its neighbour and to the node two hops away.

    Gives multiple genuinely distinct paths between most pairs — without that, routing choice is vacuous and
    the action space has nothing to express.
    """
    if node_count < 5:
        # Below five, the offset-2 chord wraps onto a node the offset-1 ring already connects, so the two
        # capacity tiers collide and a link's capacity depends on construction order.
        raise TopologyError(
            "A ring-with-chords mesh needs at least 5 nodes; below that the chords wrap onto the ring.",
            {"name": name, "node_count": node_count},
        )
    nodes = tuple(f"n{index}" for index in range(node_count))
    links: list[Link] = []
    for index, node in enumerate(nodes):
        for offset in (1, 2):
            neighbour = nodes[(index + offset) % node_count]
            if neighbour == node:
                continue
            # Capacity falls off on the chord links, so paths are not interchangeable.
            scale = 1.0 if offset == 1 else 0.6
            links.append(Link(f"{node}->{neighbour}", node, neighbour, capacity_mbps * scale))
            links.append(Link(f"{neighbour}->{node}", neighbour, node, capacity_mbps * scale))
    unique = {link.link_id: link for link in links}
    return Topology(name=name, nodes=nodes, links=tuple(unique.values()))


def _line(name: str, node_count: int, capacity_mbps: float) -> Topology:
    """A chain. One path between any pair — useful for testing allocation without routing choice."""
    nodes = tuple(f"n{index}" for index in range(node_count))
    links: list[Link] = []
    for index in range(node_count - 1):
        left, right = nodes[index], nodes[index + 1]
        links.append(Link(f"{left}->{right}", left, right, capacity_mbps))
        links.append(Link(f"{right}->{left}", right, left, capacity_mbps))
    return Topology(name=name, nodes=nodes, links=tuple(links))


_BUILDERS = {
    "mesh-6": lambda: _mesh("mesh-6", 6, 100.0),
    "mesh-10": lambda: _mesh("mesh-10", 10, 100.0),
    "line-4": lambda: _line("line-4", 4, 100.0),
}


def build_topology(name: str) -> Topology:
    """Construct a named topology.

    Named rather than parameterised so that a run config records a topology by a name that means exactly one
    thing, forever.
    """
    builder = _BUILDERS.get(name)
    if builder is None:
        raise TopologyError(f"Unknown topology {name!r}.", {"available": sorted(_BUILDERS)})
    return builder()


def available_topologies() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))
