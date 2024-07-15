import numpy as np
from topology.vertex import Vertex
from topology.edge import Edge
from topology.operations.edge_operations import edge_with_same_geometry_q
from topology.operations.edge_operations import edge_strong_equality_q
from topology.operations.edge_operations import edge_edge_intersection
from topology.operations.vertex_operations import vertex_with_same_geometry_q
from topology.operations.vertex_operations import vertex_strong_equality_q


def test_edge_with_same_geometry_q():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))

    v2: Vertex = Vertex(2, np.array([0.0, 0.0, 0.0]))
    v3: Vertex = Vertex(3, np.array([0.5, 0.5, 0.5]))

    v4: Vertex = Vertex(4, np.array([0.8, 0.8, 0.8]))
    v5: Vertex = Vertex(5, np.array([0.5, 0.5, 0.5]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v2, v3]))
    e2: Edge = Edge(2, np.array([v4, v5]))
    assert edge_with_same_geometry_q(e0, e1)
    assert not edge_with_same_geometry_q(e0, e2)
    assert not edge_with_same_geometry_q(e1, e2)


def test_edge_strong_equality_q():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))

    v2: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v3: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))

    v4: Vertex = Vertex(2, np.array([0.8, 0.8, 0.8]))
    v5: Vertex = Vertex(3, np.array([0.5, 0.5, 0.5]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(0, np.array([v2, v3]))
    e2: Edge = Edge(2, np.array([v4, v5]))
    assert edge_strong_equality_q(e0, e1)
    assert not edge_strong_equality_q(e0, e2)
    assert not edge_strong_equality_q(e1, e2)


def test_edge_edge_intersection():

    # case 1: no intersection
    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.5]))
    v1: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))

    v2: Vertex = Vertex(2, np.array([0.0, 0.0, 0.0]))
    v3: Vertex = Vertex(3, np.array([0.5, 0.5, 0.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v2, v3]))

    # No intersection
    out_0 = edge_edge_intersection(e0, e1, tag=2)
    out_1 = edge_edge_intersection(e1, e0, tag=2)
    # Intersection is commutative
    assert out_0 == out_1
    assert out_0 is None

    # case 2: Intersection in boundary by common Vertex
    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))

    v2: Vertex = Vertex(2, np.array([0.0, 0.0, 0.0]))
    v3: Vertex = Vertex(3, np.array([0.5, 0.5, 0.0]))

    e0: Edge = Edge(0, np.array([v0, v1]))
    e1: Edge = Edge(1, np.array([v2, v3]))

    # No intersection
    out_0 = edge_edge_intersection(e0, e1, tag=2)
    out_1 = edge_edge_intersection(e1, e0, tag=2)
    # Intersection is commutative
    assert vertex_strong_equality_q(out_0, out_1)
    assert not vertex_strong_equality_q(out_0, v0)
    assert not vertex_strong_equality_q(out_0, v2)
    assert vertex_with_same_geometry_q(out_0, v0)
    assert vertex_with_same_geometry_q(out_1, v2)
