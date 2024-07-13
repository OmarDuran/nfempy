import numpy as np
from topology.vertex import Vertex
from topology.edge import Edge
from topology.operations.vertex_operations import vertex_with_same_geometry_q
from topology.operations.vertex_operations import vertex_strong_equality_q
from topology.operations.vertex_operations import vertex_vertex_intersection
from topology.operations.vertex_operations import vertex_edge_intersection


def test_vertex_with_same_geometry():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.0, 0.0, 0.0]))
    assert vertex_with_same_geometry_q(v0, v1)

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([1.0e-9, 0.0, 0.0]))
    assert not vertex_with_same_geometry_q(v0, v1)


def test_vertex_strong_equality():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.0, 0.0, 0.0]))
    assert vertex_strong_equality_q(v0, v0)
    assert not vertex_strong_equality_q(v0, v1)

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(0, np.array([1.0e-9, 0.0, 0.0]))
    assert v0 == v1  # weak equality
    assert not vertex_strong_equality_q(v0, v1)


def test_vertex_vertex_intersection():
    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([1.0, 0.0, 0.0]))
    v2: Vertex = Vertex(2, np.array([0.0, 0.0, 0.0]))

    out = vertex_vertex_intersection(v1, v0, tag=3)
    assert out is None

    out = vertex_vertex_intersection(v0, v2, tag=2)
    assert vertex_strong_equality_q(out, v2)
    out = vertex_vertex_intersection(v0, v2, tag=3)
    assert not vertex_strong_equality_q(out, v2)

    v3: Vertex = Vertex(1, np.array([0.1, 0.0, 0.0]))
    v4: Vertex = Vertex(2, np.array([0.0, 0.1, 0.0]))
    out = vertex_vertex_intersection(v4, v3, tag=2, eps=0.11)
    assert vertex_strong_equality_q(v4, out)

    v3: Vertex = Vertex(1, np.array([0.1, 0.0, 0.0]))
    v4: Vertex = Vertex(2, np.array([0.0, 0.1, 0.0]))
    out = vertex_vertex_intersection(v4, v3, tag=4)
    assert out is None


def test_vertex_edge_intersection():

    v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
    v1: Vertex = Vertex(1, np.array([0.5, 0.5, 0.5]))
    v2: Vertex = Vertex(2, np.array([0.5, 0.5, 0.5]))
    v3: Vertex = Vertex(2, np.array([0.5, 0.5, 0.0]))
    v4: Vertex = Vertex(2, np.array([0.25, 0.25, 0.25]))

    # Intersecting on boundary
    e0: Edge = Edge(0, np.array([v0, v1]))
    out = vertex_edge_intersection(v2, e0)
    assert out is not None
    assert out != v2
    assert vertex_with_same_geometry_q(out, v2)

    # No intersection with boundary
    out = vertex_edge_intersection(v3, e0)
    assert out is None

    # Intersecting in edge
    out = vertex_edge_intersection(v4, e0)
    assert out is not None
    assert out != v0
    assert out != v1
    assert vertex_with_same_geometry_q(out, v4)
