import pytest
import numpy as np
from functools import partial
from geometry.operations.polygon_geometry_operations import polygon_normal
from geometry.operations.polygon_geometry_operations import convex_q
from geometry.operations.polygon_geometry_operations import triangulate_convex_polygon
from geometry.operations.polygon_geometry_operations import triangulate_polygon
from geometry.operations.polygon_geometry_operations import (
    __projection_directions,
    winding_number,
)
from geometry.operations.polygon_geometry_operations import (
    triangle_triangle_intersection,
)


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


def test_polygon_normal():

    ref_normal = np.array([0.70710678, 0.5, -0.5])
    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, -0.951057, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    assert np.all(np.isclose(polygon_normal(points), ref_normal))

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    assert np.all(np.isclose(polygon_normal(points), ref_normal))

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    assert np.all(np.isclose(polygon_normal(points), ref_normal))


def test_convex_q():
    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, -0.951057, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    assert not convex_q(points)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    assert convex_q(points)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    assert convex_q(points)


def test_triangulate_convex_polygon():
    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    triangles = triangulate_convex_polygon(points)
    assert triangles.shape == (len(points), 3, 3)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    triangles = triangulate_convex_polygon(points)
    assert triangles.shape == (len(points), 3, 3)


def test_triangulate_polygon():
    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    points = np.array(
        [
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-2.0, -2.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    triangles = triangulate_polygon(points)
    assert triangles.shape == (4, 3, 3)
    dir_idx = __projection_directions(points)
    triangle_xc = np.mean(triangles[:, :, dir_idx], axis=1)
    polygon_winding = partial(winding_number, polygon_points=points[:, dir_idx])
    valid_triangles = np.argwhere(list(map(polygon_winding, triangle_xc))).ravel()
    assert len(valid_triangles) == 4

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
            [1.0, 0.0, 0.0],
            [0.309017, -0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    triangles = triangulate_polygon(points)
    assert triangles.shape == (len(points), 3, 3)

    points = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [-0.809017, 0.587785, 0.0],
            [0.309017, 0.951057, 0.0],
        ]
    )
    points = __transform_points(points, trans_matrix)
    triangles = triangulate_polygon(points)
    assert triangles.shape == (len(points), 3, 3)
