from __future__ import annotations

import pytest

from nexus_errors.simulation_error import TopologyError
from nexus_sim.topology import Link, Topology, available_topologies, build_topology


def test_named_topologies_build() -> None:
    for name in available_topologies():
        topology = build_topology(name)
        assert topology.nodes
        assert topology.links


def test_unknown_topology_lists_what_is_available() -> None:
    with pytest.raises(TopologyError) as excinfo:
        build_topology("mesh-999")

    assert "available" in excinfo.value.context


def test_link_ordering_is_stable_across_builds() -> None:
    # Element ordering is part of the observation contract — index 7 must mean the same link every run.
    assert build_topology("mesh-6").link_ids == build_topology("mesh-6").link_ids


def test_topology_without_nodes_is_rejected() -> None:
    with pytest.raises(TopologyError):
        Topology(name="empty", nodes=(), links=())


def test_link_referencing_an_unknown_node_is_rejected() -> None:
    with pytest.raises(TopologyError):
        Topology(name="bad", nodes=("a",), links=(Link("a->ghost", "a", "ghost", 10.0),))


def test_mesh_offers_multiple_distinct_paths() -> None:
    topology = build_topology("mesh-6")

    paths = topology.k_shortest_paths("n0", "n3", k=3)

    assert len(paths) >= 2
    assert len(set(paths)) == len(paths)


def test_paths_come_out_shortest_hop_first() -> None:
    topology = build_topology("mesh-6")

    paths = topology.k_shortest_paths("n0", "n4", k=4)

    hop_counts = [len(path) for path in paths]
    assert hop_counts == sorted(hop_counts)


def test_path_enumeration_is_stable_so_the_action_space_means_one_thing() -> None:
    first = build_topology("mesh-6").k_shortest_paths("n0", "n3", k=3)
    second = build_topology("mesh-6").k_shortest_paths("n0", "n3", k=3)

    assert first == second


def test_paths_are_simple_with_no_repeated_link() -> None:
    topology = build_topology("mesh-6")

    for path in topology.k_shortest_paths("n0", "n3", k=3):
        assert len(set(path)) == len(path)


def test_line_topology_offers_exactly_one_path() -> None:
    topology = build_topology("line-4")

    assert len(topology.k_shortest_paths("n0", "n3", k=5)) == 1


def test_path_from_a_node_to_itself_is_rejected() -> None:
    with pytest.raises(TopologyError):
        build_topology("mesh-6").k_shortest_paths("n0", "n0", k=1)


def test_links_incident_to_a_node_covers_both_directions() -> None:
    topology = build_topology("line-4")

    incident = topology.links_incident_to("n1")

    assert "n0->n1" in incident
    assert "n1->n0" in incident


def test_capacities_are_returned_in_link_order() -> None:
    topology = build_topology("mesh-6")

    capacities = topology.capacities_mbps()

    assert capacities.shape == (len(topology.links),)
    assert all(value > 0.0 for value in capacities)
