import pytest
import numpy as np
from functools import partial
from topology.point_polygon_incidence import coplanar_measurement
from topology.point_polygon_incidence import point_triangle_incidence
from topology.point_polygon_incidence import point_triangle_intersection
from topology.point_polygon_incidence import points_triangle_intersection
from topology.point_polygon_incidence import point_polygon_intersection
from topology.point_polygon_incidence import points_polygon_intersection


from topology.polygon_operations import convex_q
from topology.polygon_operations import triangulate_convex_polygon
from topology.polygon_operations import triangulate_polygon
from topology.polygon_operations import __projection_directions, winding_number

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

    points_base = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0/3.0, 1.0/3.0, 0.0],
        [-1.0/3.0, -1.0/3.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0,1,2])
    point_idx = np.array([3,4,5,6,7])

    # coplanar setting
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        measurement = coplanar_measurement(points[idx], a, b ,c)
        assert np.isclose(measurement, 0.0)

    # no coplanar setting
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        measurement = coplanar_measurement(points_base[idx], a, b ,c)
        assert not np.isclose(measurement, 0.0)

def test_point_triangle_incidence():

    points_base = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0/3.0, 1.0/3.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ])

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0,1,2])
    point_idx = np.array([3,4,5,6,7,8,9])

    # incident points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_incidence(points[idx], a, b ,c)
        assert output

    # no incident points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_incidence(points_base[idx], a, b ,c)
        assert not output

def test_point_triangle_intersection():

    points_base = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0/3.0, 1.0/3.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ])

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0,1,2])
    point_idx = np.array([3,4,5,6,7,8,9])

    # intersecting points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_intersection(points[idx], a, b ,c)
        assert np.all(np.isclose(output, points[idx]))

    # no intersecting points
    a, b, c = points[triangle_idx]
    for idx in point_idx:
        output = point_triangle_intersection(points_base[idx], a, b ,c)
        assert output is None

def test_points_triangle_intersection():

    points_base = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0/3.0, 1.0/3.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ])

    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    points = __transform_points(points, trans_matrix)

    triangle_idx = np.array([0,1,2])
    point_idx = np.array([3,4,5,6,7,8,9])

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
    polygon_points_base = np.array([
        [-0.809017, -0.587785, 0.0],
        [-0.809017, 0.587785, 0.0],
        [0.309017, 0.951057, 0.0],
        [1., 0., 0.0],
        [0.309017, -0.951057, 0.0],
    ])
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array([
        [-0.809017, -0.587785, 0.0],
        [-0.809017, 0.587785, 0.0],
        [0.309017, 0.951057, 0.0],
        [1., 0., 0.0],
        [0.309017, -0.951057, 0.0],
        [0.0, 0.0, 0.0],
    ])
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)


    for p in points:
        output = point_polygon_intersection(p, polygon_points)
        assert np.all(np.isclose(output, p))

    for p in points_base:
        output = point_polygon_intersection(p, polygon_points)
        assert output is None


    # no convex
    polygon_points_base = np.array([
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [-2.0, 2.0, 0.0],
        [-2.0, -2.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array([
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
    ])
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
    polygon_points_base = np.array([
        [-0.809017, -0.587785, 0.0],
        [-0.809017, 0.587785, 0.0],
        [0.309017, 0.951057, 0.0],
        [1., 0., 0.0],
        [0.309017, -0.951057, 0.0],
    ])
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array([
        [-0.809017, -0.587785, 0.0],
        [-0.809017, 0.587785, 0.0],
        [0.309017, 0.951057, 0.0],
        [1., 0., 0.0],
        [0.309017, -0.951057, 0.0],
        [0.0, 0.0, 0.0],
    ])
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    output, intx_q = points_polygon_intersection(points, polygon_points)
    assert np.all(np.isclose(output, points))
    output, intx_q = points_polygon_intersection(points_base, polygon_points)
    assert output.shape[0] == 0


    # no convex
    polygon_points_base = np.array([
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [-2.0, 2.0, 0.0],
        [-2.0, -2.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    polygon_points = polygon_points_base.copy()
    polygon_points = __transform_points(polygon_points, trans_matrix)

    points_base = np.array([
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
    ])
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    output, intx_q = points_polygon_intersection(points, polygon_points)
    assert np.all(np.isclose(output, points))
    output, intx_q = points_polygon_intersection(points_base, polygon_points)
    assert output.shape[0] == 0


