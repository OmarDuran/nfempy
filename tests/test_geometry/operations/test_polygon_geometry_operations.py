import pytest
import numpy as np
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


def test_triangle_triangle_intersection():

    triangle_object = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    # out = triangle_triangle_intersection(triangle_object, triangle_tool)
    # assert np.all(np.isclose(out,np.array([[[0., 0., 0.]]])))

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    out = triangle_triangle_intersection(triangle_object, triangle_tool)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])))

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    out = triangle_triangle_intersection(triangle_object, triangle_tool)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])))

    triangle_tool = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    out = triangle_triangle_intersection(triangle_object, triangle_tool)
    assert np.all(np.isclose(out, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])))

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    out = triangle_triangle_intersection(triangle_object, triangle_tool)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])))

    triangle_tool = np.array(
        [
            [0.3, 0.3, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    out = triangle_triangle_intersection(triangle_object, triangle_tool)
    assert np.all(np.isclose(out, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])))
