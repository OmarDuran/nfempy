import pytest
import numpy as np
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell

from topology.line_line_incidence import line_line_plot
from topology.line_line_incidence import line_line_intersection
from topology.line_line_incidence import lines_line_intersection
from topology.line_line_incidence import lines_lines_intersection
from topology.edge_operations import edge_with_same_geometry_q
from topology.edge_operations import edge_strong_equality_q
from topology.edge_operations import edge_edge_intersection
from topology.vertex_operations import vertex_with_same_geometry_q
from topology.vertex_operations import vertex_strong_equality_q


def transformation_matrix(theta, tx, ty, tz):
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return np.dot(T, Rz)


def transform_points(points, transformation_matrix):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points_homogeneous = np.dot(points_homogeneous, transformation_matrix.T)
    transformed_points = transformed_points_homogeneous[:, :3]
    return transformed_points


def test_line_line_intersection():

    a = np.array([0.3, 0.2, 0.0])
    b = np.array([0.3, 0.4, 0.0])
    c = np.array([0.1, 0.1, 0.0])
    d = np.array([0.2, -0.1, 0.0])
    out = line_line_intersection(a, b, c, d)
    assert out is None

    a = np.array([0.2, 0.25, 0.0])
    b = np.array([0.4, 0.25, 0.0])
    c = np.array([0.1, 0.1, 0.0])
    d = np.array([0.2, -0.1, 0.0])
    out = line_line_intersection(a, b, c, d)
    assert out is None

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 1.0, 1.0])
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[0.5, 0.5, 0.5]])))

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = transformation_matrix(theta, tx, ty, tz)

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 1.0, 1.0])
    points = np.array([a, b, c, d])
    transformed_points = transform_points(points, trans_matrix)
    a, b, d, c = transformed_points
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[1.0, 2.70710678, 3.5]])))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, -1.0e-3, 0.0])
    d = np.array([0.0, 1.0, 0.0])
    points = np.array([a, b, c, d])
    transformed_points = transform_points(points, trans_matrix)
    a, b, d, c = transformed_points
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[1.0, 2.0, 3.0]])))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    c = np.array([0.5, 0.5, 0.5])
    d = np.array([1.5, 1.5, 1.5])
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[1.0, 1.0, 1.0]])))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    c = np.array([-1.5, -1.5, -1.5])
    d = np.array([0.5, 0.5, 0.5])
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0]])))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    c = np.array([-1.5, -1.5, -1.5])
    d = np.array([1.5, 1.5, 1.5])
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    c = np.array([-1.5, -1.5, -1.5])
    d = np.array([0.0, 0.0, 0.0])
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0]])))

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([-1.5, -1.5, -1.5])
    c = np.array([-1.5, -1.5, -1.5])
    d = np.array([0.0, 0.0, 0.0])
    out = line_line_intersection(a, b, c, d)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0], [-1.5, -1.5, -1.5]])))


def test_lines_line_intersection():

    # object
    a = np.array([0.1, 0.1, 0.0])
    b = np.array([0.4, 0.4, 0.0])

    # tools
    a0 = np.array([0.1, 0.1, 0.0])
    b0 = np.array([0.2, -0.1, 0.0])
    a1 = np.array([0.3, 0.2, 0.0])
    b1 = np.array([0.3, 0.4, 0.0])
    a2 = np.array([0.2, 0.25, 0.0])
    b2 = np.array([0.4, 0.25, 0.0])

    lines = np.array([[a0, b0], [a1, b1], [a2, b2]])
    out = lines_line_intersection(lines, a, b)
    assert np.all(
        np.isclose(
            out, np.array([[[0.1, 0.1, 0.0]], [[0.3, 0.3, 0.0]], [[0.25, 0.25, 0.0]]])
        )
    )

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = transformation_matrix(theta, tx, ty, tz)
    points = np.array([a, b, a0, b0, a1, b1, a2, b2])
    transformed_points = transform_points(points, trans_matrix)
    a, b, a0, b0, a1, b1, a2, b2 = transformed_points
    lines = np.array([[a0, b0], [a1, b1], [a2, b2]])
    out = lines_line_intersection(lines, a, b)
    assert np.all(
        np.isclose(
            out,
            np.array(
                [
                    [[1.0, 2.14142136, 3.0]],
                    [[1.0, 2.42426407, 3.0]],
                    [[1.0, 2.35355339, 3.0]],
                ]
            ),
        )
    )


def test_lines_lines_intersection():

    # tools and objects
    a0 = np.array([0.1, 0.1, 0.0])
    b0 = np.array([0.4, 0.4, 0.0])
    a1 = np.array([0.1, 0.1, 0.0])
    b1 = np.array([0.2, -0.1, 0.0])
    a2 = np.array([0.3, 0.2, 0.0])
    b2 = np.array([0.3, 0.4, 0.0])
    a3 = np.array([0.2, 0.25, 0.0])
    b3 = np.array([0.4, 0.25, 0.0])

    lines = np.array([[a0, b0], [a1, b1], [a2, b2], [a3, b3]])
    out = lines_lines_intersection(lines, lines)
    assert np.all(
        np.isclose(
            out[0],
            np.array([[[0.1, 0.1, 0.0]], [[0.3, 0.3, 0.0]], [[0.25, 0.25, 0.0]]]),
        )
    )
    assert np.all(np.isclose(out[1], np.array([[[0.1, 0.1, 0.0]]])))
    assert np.all(np.isclose(out[2], np.array([[[0.3, 0.3, 0.0]], [[0.3, 0.25, 0.0]]])))
    assert np.all(
        np.isclose(out[3], np.array([[[0.25, 0.25, 0.0]], [[0.3, 0.25, 0.0]]]))
    )

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = transformation_matrix(theta, tx, ty, tz)
    points = np.array([a0, b0, a1, b1, a2, b2, a3, b3])
    transformed_points = transform_points(points, trans_matrix)
    a0, b0, a1, b1, a2, b2, a3, b3 = transformed_points
    lines = np.array([[a0, b0], [a1, b1], [a2, b2], [a3, b3]])
    out = lines_lines_intersection(lines, lines)
    assert np.all(
        np.isclose(
            out[0],
            np.array(
                [
                    [[1.0, 2.14142136, 3.0]],
                    [[1.0, 2.42426407, 3.0]],
                    [[1.0, 2.35355339, 3.0]],
                ]
            ),
        )
    )
    assert np.all(
        np.isclose(
            out[1],
            np.array([[[1.0, 2.14142136, 3.0]]]),
        )
    )
    assert np.all(
        np.isclose(
            out[2],
            np.array([[[1.0, 2.42426407, 3.0]], [[1.03535534, 2.38890873, 3.0]]]),
        )
    )
    assert np.all(
        np.isclose(
            out[3],
            np.array([[[1.0, 2.35355339, 3.0]], [[1.03535534, 2.38890873, 3.0]]]),
        )
    )


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
