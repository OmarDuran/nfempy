import pytest
import numpy as np
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell

from topology.point_line_incidence import colinear_measurement
from topology.point_line_incidence import point_line_incidence
from topology.point_line_incidence import point_line_intersection
from topology.point_line_incidence import points_line_intersection
from topology.point_line_incidence import points_line_argsort

from topology.vertex_operations import vertex_with_same_geometry_q
from topology.vertex_operations import vertex_strong_equality_q
from topology.vertex_operations import vertex_vertex_intersection
from topology.vertex_operations import vertex_edge_intersection


def test_point_line_colinear_measurement():

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    gamma = 0.5
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    measurement = colinear_measurement(c, a, b)
    assert np.isclose(measurement, 0.0)

    gamma = 1.0e-15
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    measurement = colinear_measurement(c, a, b)
    assert np.isclose(measurement, 0.0)

    gamma = 1.0e8
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    measurement = colinear_measurement(c, a, b)
    assert np.isclose(measurement, 0.0)

    gamma = -1.0e8
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    measurement = colinear_measurement(c, a, b)
    assert np.isclose(measurement, 0.0)

    a = np.array([-1.0, 0.0, -1.0])
    b = np.array([1.0, 10.0, 0.0])
    c = np.array([1.0, 1.0, 1.0])
    measurement = colinear_measurement(c, a, b)
    assert np.isclose(measurement, 13.124404748406688)


def test_point_line_incidence():

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    gamma = 0.0
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    assert point_line_incidence(c, a, b)

    gamma = 1.0e15
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    assert point_line_incidence(c, a, b)

    gamma = 1.0e8
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    assert point_line_incidence(c, a, b)

    a = np.array([-1.0, 0.0, -1.0])
    b = np.array([1.0, 10.0, 0.0])
    c = np.array([1.0, 1.0, 1.0])
    assert not point_line_incidence(c, a, b)


def test_point_line_intersection():

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    gamma = 0.0
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    p = point_line_intersection(c, a, b)
    assert np.all(np.isclose(p, b))

    gamma = 1.0
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    p = point_line_intersection(c, a, b)
    assert np.all(np.isclose(p, a))

    gamma = 0.5
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    p = point_line_intersection(c, a, b)
    assert np.all(np.isclose(p, c))

    gamma = 0.0 - 1.0e-9
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    p = point_line_intersection(c, a, b)
    assert p is None

    gamma = 1.0 + 1.0e-9
    c = (1 - gamma) * np.array([1.0, 1.0, 1.0])
    p = point_line_intersection(c, a, b)
    assert p is None

    a = np.array([-1.0, 0.0, -1.0])
    b = np.array([1.0, 10.0, 0.0])
    c = np.array([1.0, 1.0, 1.0])
    p = point_line_intersection(c, a, b)
    assert p is None


def test_points_line_intersection():

    # points for segment
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    gammas = np.linspace(-1.0, 1.1, 5)
    points = np.array([(1 - gamma) * np.array([1.0, 1.0, 1.0]) for gamma in gammas])
    out, intx_q = points_line_intersection(points, a, b)
    assert np.all(np.isclose(out[0], np.array([0.95, 0.95, 0.95])))
    assert np.all(np.isclose(out[1], np.array([0.425, 0.425, 0.425])))

def test_points_line_argsort():

    # points for segment
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    gammas = np.linspace(-1.0, 1.1, 5)
    points = np.array([(1 - gamma) * np.array([1.0, 1.0, 1.0]) for gamma in gammas])
    # a to b oriented
    out = points_line_argsort(points, a, b, ba_sorting= False)
    assert np.all(np.isclose(out, np.array([4,3,2,1,0])))

    # b to a oriented
    out = points_line_argsort(points, a, b, ba_sorting=True)
    assert np.all(np.isclose(out, np.array([2, 1, 3, 0, 4])))


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
