import numpy as np

from geometry.operations.point_geometry_operations import colinear_measurement
from geometry.operations.point_geometry_operations import point_line_incidence
from geometry.operations.point_geometry_operations import point_line_intersection
from geometry.operations.point_geometry_operations import points_line_intersection
from geometry.operations.point_geometry_operations import points_line_argsort

from geometry.operations.point_geometry_operations import coplanar_measurement
from geometry.operations.point_geometry_operations import point_triangle_incidence
from geometry.operations.point_geometry_operations import point_triangle_intersection
from geometry.operations.point_geometry_operations import points_triangle_intersection
from geometry.operations.point_geometry_operations import point_polygon_intersection
from geometry.operations.point_geometry_operations import points_polygon_intersection



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
    out = points_line_argsort(points, a, b, ba_sorting=False)
    assert np.all(np.isclose(out, np.array([4, 3, 2, 1, 0])))

    # b to a oriented
    out = points_line_argsort(points, a, b, ba_sorting=True)
    assert np.all(np.isclose(out, np.array([2, 1, 3, 0, 4])))

def __transformation_matrix(theta, tx, ty, tz):
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])
    return T @ Rx @ Ry @ Rz


def __transform_points(points, transformation_matrix):
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points_homogeneous = np.dot(points_homogeneous, transformation_matrix.T)
    transformed_points = transformed_points_homogeneous[:, :3]
    return transformed_points


def test_coplanar_measurement():

    points_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [-1.0 / 3.0, -1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0, 1, 2])
    point_idx = np.array([3, 4, 5, 6, 7])

    # coplanar setting
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        measurement = coplanar_measurement(points[idx], a, b, c)
        assert np.isclose(measurement, 0.0)

    # no coplanar setting
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        measurement = coplanar_measurement(points_base[idx], a, b, c)
        assert not np.isclose(measurement, 0.0)


def test_point_triangle_incidence():

    points_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0, 1, 2])
    point_idx = np.array([3, 4, 5, 6, 7, 8, 9])

    # incident points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_incidence(points[idx], a, b, c)
        assert output

    # no incident points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_incidence(points_base[idx], a, b, c)
        assert not output


def test_point_triangle_intersection():

    points_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0, 1, 2])
    point_idx = np.array([3, 4, 5, 6, 7, 8, 9])

    # intersecting points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_intersection(points[idx], a, b, c)
        assert np.all(np.isclose(output, points[idx]))

    # no intersecting points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_intersection(points_base[idx], a, b, c)
        assert output is None


def test_points_triangle_intersection():

    points_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0, 1, 2])
    point_idx = np.array([3, 4, 5, 6, 7, 8, 9])

    # intersecting points
    a, b, c = points[triangle_idx]
    output, intx_q = points_triangle_intersection(points[point_idx], a, b, c)
    assert np.all(np.isclose(output, points[point_idx]))

    # no intersecting points
    a, b, c = points[triangle_idx]
    output, intx_q = points_triangle_intersection(points_base[point_idx], a, b, c)
    assert output.shape[0] == 0


def test_point_polygon_intersection():

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    # convex
    polygon_points_base = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
        ]
    )
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    for p in points:
        output = point_polygon_intersection(p, polygon_points)
        assert np.all(np.isclose(output, p))

    for p in points_base:
        output = point_polygon_intersection(p, polygon_points)
        assert output is None

    # no convex
    polygon_points_base = np.array(
        [
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array(
        [
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
    )
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    for p in points:
        output = point_polygon_intersection(p, polygon_points)
        assert np.all(np.isclose(output, p))

    for p in points_base:
        output = point_polygon_intersection(p, polygon_points)
        assert output is None


def test_points_polygon_intersection():

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    # convex
    polygon_points_base = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
        ]
    )
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    output, intx_q = points_polygon_intersection(points, polygon_points)
    assert np.all(np.isclose(output, points))
    output, intx_q = points_polygon_intersection(points_base, polygon_points)
    assert output.shape[0] == 0

    # no convex
    polygon_points_base = np.array(
        [
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array(
        [
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ]
    )
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    output, intx_q = points_polygon_intersection(points, polygon_points)
    assert np.all(np.isclose(output, points))
    output, intx_q = points_polygon_intersection(points_base, polygon_points)
    assert output.shape[0] == 0
