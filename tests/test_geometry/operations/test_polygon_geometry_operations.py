import pytest
import numpy as np
from geometry.operations.polygon_geometry_operations import triangle_triangle_intersection
from geometry.operations.polygon_geometry_operations import triangle_polygon_intersection


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

    theta = np.pi / 6
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

    triangle_object_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle_object = triangle_object_base.copy()
    triangle_object = __transform_points(triangle_object, trans_matrix)

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)

    out = triangle_triangle_intersection(triangle_tool, triangle_object)
    assert np.all(np.isclose(out, np.array([[[1.0, 2.0, 3.0]]])))

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_triangle_intersection(triangle_tool, triangle_object)
    assert np.all(
        np.isclose(out, np.array([[1.0, 2.0, 3.0], [1.75, 2.21650635, 3.625]]))
    )

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_triangle_intersection(triangle_tool, triangle_object)
    assert np.all(
        np.isclose(out, np.array([[0.5669873, 2.875, 3.21650635], [1.0, 2.0, 3.0]]))
    )

    triangle_tool = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_triangle_intersection(triangle_tool, triangle_object)
    assert np.all(
        np.isclose(
            out, np.array([[0.5669873, 2.875, 3.21650635], [1.75, 2.21650635, 3.625]])
        )
    )

    triangle_tool = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_triangle_intersection(triangle_tool, triangle_object)
    assert np.all(
        np.isclose(
            out, np.array([[1.0, 2.0, 3.0], [1.15849365, 2.54575318, 3.42075318]])
        )
    )

    triangle_tool = np.array(
        [
            [0.3, 0.3, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_triangle_intersection(triangle_tool, triangle_object)
    assert np.all(
        np.isclose(
            out, np.array([[1.0, 2.0, 3.0], [1.09509619, 2.32745191, 3.25245191]])
        )
    )

def test_triangle_polygon_intersection():

    theta = -np.pi / 6
    tx, ty, tz = 0.1, 0.2, 0.3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)

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

    triangle_tool = np.array(
        [
            [0.3, 0.6, 0.1],
            [0.3, 0.4, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_polygon_intersection(triangle_tool, polygon_points)
    assert np.all(np.isclose(out, np.array([[0.49820508, 0.25514428, 0.00269238]]) ))

    triangle_tool = np.array(
        [
            [0.3, 0.6, 0.0],
            [0.3, 0.4, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_polygon_intersection(triangle_tool,polygon_points)
    assert np.all(np.isclose(out, np.array([[ 0.49820508,  0.25514428,  0.00269238],
       [ 0.58480762,  0.38014428, -0.12721143]]) ))

    triangle_tool = np.array(
        [
            [0.3, 0.6, 0.0],
            [0.3, -1.6, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_polygon_intersection(triangle_tool,polygon_points)
    assert np.all(np.isclose(out, np.array([[ 0.58480762,  0.38014428, -0.12721143],
       [-0.08555112, -0.58743521,  0.87832668]]) ))

    triangle_tool = np.array(
        [
            [-0.809017, -0.587785, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_polygon_intersection(triangle_tool,polygon_points)
    assert np.all(np.isclose(out, np.array([[ 0.85      , -0.44951905,  0.175     ],
       [-0.76128112,  0.35810633,  0.78290468]]) ))

    triangle_tool = np.array(
        [
            [-0.809017, -0.587785, -0.2],
            [1.0, 0.0, -0.2],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_polygon_intersection(triangle_tool,polygon_points)
    assert np.all(np.isclose(out, np.array([[-0.54556548,  0.43592194,  0.59416739],
       [ 0.79716878, -0.23709921,  0.08758016]]) ))


    triangle_tool = np.array(
        [
            [-0.809017, -0.587785, 0.2],
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 1.0],
        ]
    )
    triangle_tool = __transform_points(triangle_tool, trans_matrix)
    out = triangle_polygon_intersection(triangle_tool,polygon_points)
    assert out is None