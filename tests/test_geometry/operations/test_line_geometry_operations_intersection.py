import numpy as np
from geometry.operations.line_geometry_operations import line_line_intersection
from geometry.operations.line_geometry_operations import lines_line_intersection
from geometry.operations.line_geometry_operations import lines_lines_intersection
from geometry.operations.line_geometry_operations import line_triangle_intersection
from geometry.operations.line_geometry_operations import lines_triangle_intersection
from geometry.operations.line_geometry_operations import line_polygon_intersection
from geometry.operations.line_geometry_operations import lines_polygon_intersection


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
    out, _ = line_line_intersection(a, b, c, d)
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
    out, _ = line_line_intersection(a, b, c, d)  # bc intersection
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


def test_line_triangle_intersection():
    triangle_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle = triangle_base.copy()

    points_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, -0.5],
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    triangle = __transform_points(triangle, trans_matrix)
    points = __transform_points(points, trans_matrix)

    a, b, c = triangle
    overlap_intersection_lines_idxs = np.array([[0, 1], [1, 2], [2, 0]])
    for lines_idx in overlap_intersection_lines_idxs:
        line = points[lines_idx]
        out = line_triangle_intersection(line, a, b, c)
        assert np.all(np.isclose(out, line))

    no_intersection_lines_idxs = np.array([[6, 8], [7, 9], [8, 10], [9, 11]])
    for lines_idx in no_intersection_lines_idxs:
        line = points[lines_idx]
        out = line_triangle_intersection(line, a, b, c)
        assert out is None

    point_intersection_line_idxs = np.array([[1, 3], [2, 3], [6, 7], [8, 9], [10, 11]])
    intx_data = np.empty((0, 3), float)
    for lines_idx in point_intersection_line_idxs:
        line = points[lines_idx]
        out = line_triangle_intersection(line, a, b, c)
        intx_data = np.append(intx_data, out, axis=0)

    ref_data = np.array(
        [
            [1.5, 2.14644661, 3.85355339],
            [0.5, 2.85355339, 3.14644661],
            [1.0, 2.0, 3.0],
            [1.0, 2.33333333, 3.33333333],
            [1.0, 2.5, 3.5],
        ]
    )
    assert np.all(np.isclose(ref_data, intx_data))

    line_intersection_lines_idxs = np.array([[3, 4], [3, 5]])
    intx_data = np.empty((0, 2, 3), float)
    for lines_idx in line_intersection_lines_idxs:
        line = points[lines_idx]
        out = line_triangle_intersection(line, a, b, c)
        intx_data = np.append(intx_data, np.array([out[:, 0, :]]), axis=0)
    ref_data = np.array(
        [
            [[1.0, 2.5, 3.5], [1.0, 2.0, 3.0]],
            [[1.3, 2.28786797, 3.71213203], [1.375, 2.10983496, 3.64016504]],
        ]
    )
    assert np.all(np.isclose(ref_data, intx_data))


def test_lines_triangle_intersection():
    triangle_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    triangle = triangle_base.copy()

    points_base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, -0.5],
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    points = points_base.copy()

    theta = np.pi / 4
    tx, ty, tz = 1, 2, 3
    trans_matrix = __transformation_matrix(theta, tx, ty, tz)
    triangle = __transform_points(triangle, trans_matrix)
    points = __transform_points(points, trans_matrix)

    a, b, c = triangle
    overlap_intersection_lines_idxs = np.array([[0, 1], [1, 2], [2, 0]])
    lines = points[overlap_intersection_lines_idxs]
    out = lines_triangle_intersection(lines, a, b, c)
    ref_data = np.array(
        [
            [[1.0, 2.0, 3.0], [1.5, 2.14644661, 3.85355339]],
            [[1.5, 2.14644661, 3.85355339], [0.5, 2.85355339, 3.14644661]],
            [[0.5, 2.85355339, 3.14644661], [1.0, 2.0, 3.0]],
        ]
    )
    assert np.all(np.isclose(out[0], ref_data))

    no_intersection_lines_idxs = np.array([[6, 8], [7, 9], [8, 10], [9, 11]])
    lines = points[no_intersection_lines_idxs]
    out = lines_triangle_intersection(lines, a, b, c)
    assert out[0].shape[0] == 0

    point_intersection_line_idxs = np.array([[1, 3], [2, 3], [6, 7], [8, 9], [10, 11]])
    lines = points[point_intersection_line_idxs]
    out = lines_triangle_intersection(lines, a, b, c)
    ref_data = np.array(
        [
            [[1.5, 2.14644661, 3.85355339]],
            [[0.5, 2.85355339, 3.14644661]],
            [[1.0, 2.0, 3.0]],
            [[1.0, 2.33333333, 3.33333333]],
            [[1.0, 2.5, 3.5]],
        ]
    )
    assert np.all(np.isclose(out[0], ref_data))

    line_intersection_lines_idxs = np.array([[3, 4], [3, 5]])
    lines = points[line_intersection_lines_idxs]
    out = lines_triangle_intersection(lines, a, b, c)
    ref_data = np.array(
        [
            [[[1.0, 2.5, 3.5]], [[1.0, 2.0, 3.0]]],
            [[[1.3, 2.28786797, 3.71213203]], [[1.375, 2.10983496, 3.64016504]]],
        ]
    )
    assert np.all(np.isclose(out[0], ref_data))


def test_line_polygon_intersection():

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
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, -0.5],
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    overlap_intersection_lines_idxs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
    for lines_idx in overlap_intersection_lines_idxs:
        line = points[lines_idx]
        out = line_polygon_intersection(line, polygon_points)
        assert np.all(np.isclose(out, line))

    no_intersection_lines_idxs = np.array(
        [[7, 8], [10, 12], [11, 13], [12, 14], [13, 15]]
    )
    for lines_idx in no_intersection_lines_idxs:
        line = points[lines_idx]
        out = line_polygon_intersection(line, polygon_points)
        assert out is None

    point_intersection_line_idxs = np.array([[11, 12], [13, 14], [15, 16]])
    intx_data = np.empty((0, 3), float)
    for lines_idx in point_intersection_line_idxs:
        line = points[lines_idx]
        out = line_polygon_intersection(line, polygon_points)
        intx_data = np.append(intx_data, out, axis=0)
    ref_data = np.array(
        [[1.0, 2.0, 3.0], [1.0, 2.33333333, 3.33333333], [1.0, 2.5, 3.5]]
    )
    assert np.all(np.isclose(ref_data, intx_data))

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

    point_intersection_line_idxs = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [5, 6], [7, 8], [9, 10]]
    )
    intx_data = np.empty((0, 3), float)
    for lines_idx in point_intersection_line_idxs:
        line = points[lines_idx]
        out = line_polygon_intersection(line, polygon_points)
        intx_data = np.append(intx_data, out, axis=0)
    ref_data = np.array(
        [
            [3.0, 0.58578644, 4.41421356],
            [1.0, 4.0, 5.0],
            [1.0, 4.0, 5.0],
            [0.5, 2.85355339, 3.14644661],
            [0.5, 2.85355339, 3.14644661],
            [-1.0, 3.41421356, 1.58578644],
            [-1.0, 3.41421356, 1.58578644],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [3.0, 0.58578644, 4.41421356],
            [1.5, 1.14644661, 2.85355339],
            [1.0, 2.0, 3.0],
            [1.33333333, 2.09763107, 3.56903559],
            [0.66666667, 1.90236893, 2.43096441],
            [1.0, 2.4, 3.4],
            [1.0, 1.6, 2.6],
        ]
    )
    assert np.all(np.isclose(ref_data, intx_data))


def test_lines_polygon_intersection():

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
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, -0.5],
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    points = points_base.copy()
    points = __transform_points(points, trans_matrix)

    overlap_intersection_lines_idxs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
    lines = points[overlap_intersection_lines_idxs]
    out = lines_polygon_intersection(lines, polygon_points)
    ref_data = np.array(
        [
            [[0.889384, 1.37981632, 2.22338168], [0.301599, 2.38322808, 2.39553992]],
            [[0.301599, 2.38322808, 2.39553992], [0.67898, 2.85703242, 3.40304158]],
            [[0.67898, 2.85703242, 3.40304158], [1.5, 2.14644661, 3.85355339]],
            [[1.5, 2.14644661, 3.85355339], [1.630037, 1.23347656, 3.12448344]],
            [[1.630037, 1.23347656, 3.12448344], [0.889384, 1.37981632, 2.22338168]],
        ]
    )
    assert np.all(np.isclose(ref_data, out[0]))

    no_intersection_lines_idxs = np.array(
        [[7, 8], [10, 12], [11, 13], [12, 14], [13, 15]]
    )
    lines = points[no_intersection_lines_idxs]
    out = lines_polygon_intersection(lines, polygon_points)
    assert out[0].shape[0] == 0

    point_intersection_line_idxs = np.array([[11, 12], [13, 14], [15, 16]])
    lines = points[point_intersection_line_idxs]
    out = lines_polygon_intersection(lines, polygon_points)
    ref_data = np.array(
        [[[1.0, 2.0, 3.0]], [[1.0, 2.33333333, 3.33333333]], [[1.0, 2.5, 3.5]]]
    )
    assert np.all(np.isclose(ref_data, out[0]))

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

    point_intersection_line_idxs = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [5, 6], [7, 8], [9, 10]]
    )
    lines = points[point_intersection_line_idxs]
    out = lines_polygon_intersection(lines, polygon_points)
    ref_data = np.array(
        [
            [[3.0, 0.58578644, 4.41421356], [1.0, 4.0, 5.0]],
            [[1.0, 4.0, 5.0], [0.5, 2.85355339, 3.14644661]],
            [[0.5, 2.85355339, 3.14644661], [-1.0, 3.41421356, 1.58578644]],
            [[-1.0, 3.41421356, 1.58578644], [1.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [3.0, 0.58578644, 4.41421356]],
            [[1.5, 1.14644661, 2.85355339], [1.0, 2.0, 3.0]],
            [
                [1.33333333, 2.09763107, 3.56903559],
                [0.66666667, 1.90236893, 2.43096441],
            ],
            [[1.0, 2.4, 3.4], [1.0, 1.6, 2.6]],
        ]
    )
    assert np.all(np.isclose(ref_data, out[0]))
