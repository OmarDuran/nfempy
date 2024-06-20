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

from topology.shape_manipulation import vertex_with_same_geometry
from topology.shape_manipulation import vertex_strong_equality
from topology.shape_manipulation import collapse_vertex
from topology.shape_manipulation import vertex_edge_boundary_intersection

from topology.graphical_shape import draw_vertex_sequence

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

    gamma = 1.0e+15
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
    out = points_line_intersection(points, a, b)
    assert out[0] is None
    assert out[1] is None
    assert np.all(np.isclose(out[2],np.array([0.95, 0.95, 0.95])))
    assert np.all(np.isclose(out[3], np.array([0.425, 0.425, 0.425])))
    assert out[4] is None



def test_vertex_with_same_geometry():

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([0.0,0.0,0.0]))
    assert vertex_with_same_geometry(v0, v1)

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([1.0e-9,0.0,0.0]))
    assert not vertex_with_same_geometry(v0, v1)

def test_vertex_strong_equality():

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([0.0,0.0,0.0]))
    assert vertex_strong_equality(v0, v0)
    assert not vertex_strong_equality(v0, v1)

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(0, np.array([1.0e-9,0.0,0.0]))
    assert v0 == v1 # weak equality
    assert not vertex_strong_equality(v0, v1)

def test_collapse_vertex():
    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([1.0,0.0,0.0]))
    v2: Vertex = Vertex(2, np.array([0.0,0.0,0.0]))

    v0 = collapse_vertex(v1, v0)
    assert not vertex_strong_equality(v0, v1)

    v2 = collapse_vertex(v0, v2)
    assert vertex_strong_equality(v0, v2)

    v3: Vertex = Vertex(1, np.array([0.1,0.0,0.0]))
    v4: Vertex = Vertex(2, np.array([0.0,0.1,0.0]))
    v3 = collapse_vertex(v4, v3, eps=0.11)
    assert vertex_strong_equality(v4, v3)

def test_vertex_tool_edge_object():

    v0: Vertex = Vertex(0, np.array([0.0,0.0,0.0]))
    v1: Vertex = Vertex(1, np.array([0.5,0.5,0.5]))
    v2: Vertex = Vertex(2, np.array([0.5,0.5,0.5]))
    v3: Vertex = Vertex(2, np.array([0.5,0.5,0.0]))

    # Intersecting on boundary
    e0: Edge = Edge(0, np.array([v0, v1]))
    out = vertex_edge_boundary_intersection(v2,e0)
    assert out is not None
    assert out != v2
    assert vertex_with_same_geometry(out,v2)

    # No intersection with boundary
    out = vertex_edge_boundary_intersection(v3,e0)
    assert out is None


