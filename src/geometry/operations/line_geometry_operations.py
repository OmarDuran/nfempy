from functools import partial
from globals import geometry_collapse_precision as collapse_precision
from globals import geometry_point_line_incidence_tol as p_incidence_tol
from globals import geometry_line_line_incidence_tol as l_incidence_tol
from globals import geometry_line_polygon_incidence_tol as s_incidence_tol

import numpy as np
import matplotlib.pyplot as plt
from geometry.operations.point_geometry_operations import points_line_argsort
from geometry.operations.point_geometry_operations import points_line_intersection
from geometry.operations.point_geometry_operations import point_triangle_intersection
from geometry.operations.polygon_operations import triangulate_polygon


def line_line_intersection(
    a: np.array, b: np.array, c: np.array, d: np.array, eps: float = l_incidence_tol
) -> float:

    out, intx_q = points_line_intersection(np.array([a, b]), c, d, p_incidence_tol)
    if len(out) > 0 and np.any(intx_q):
        if np.sum(intx_q) > 1:
            return out, intx_q
        else:
            assert np.all(np.isclose(out, np.array([a, b])[intx_q]))
            return out

    # Direction vectors
    d1 = b - a
    d2 = d - c

    # Matrix form Ax = r
    A = np.array([d1, -d2]).T
    r = c - a

    # Solve the least squares problem
    t, s = np.linalg.lstsq(A, r, rcond=None)[0]

    # Calculate the intersection points
    pa = a + t * d1
    pc = c + s * d2

    # parameters t and s should be in the unitary interval
    t_l = 0.0 < t or np.isclose(t, 0.0, rtol=eps, atol=eps)
    t_r = t < 1.0 or np.isclose(t, 1.0, rtol=eps, atol=eps)
    s_l = 0.0 < s or np.isclose(s, 0.0, rtol=eps, atol=eps)
    s_r = s < 1.0 or np.isclose(s, 1.0, rtol=eps, atol=eps)
    t_is_bounded_q = t_l and t_r
    s_is_bounded_q = s_l and s_r
    # Check if the intersection points are the same
    intx_equality = np.allclose(pa, pc, rtol=eps, atol=eps)
    if intx_equality and t_is_bounded_q and s_is_bounded_q:
        return np.array([pa])
    else:
        return None  # No intersection


def lines_line_intersection(
    lines: np.array, a: np.array, b: np.array, eps: float = l_incidence_tol
) -> float:
    # compute intersections
    line_line_intx = partial(line_line_intersection, c=a, d=b, eps=eps)
    result = [line_line_intx(line[0], line[1]) for line in lines]

    # filter lines outside segment
    result = np.array(list(filter(lambda x: x is not None, result)))
    return result


def lines_lines_intersection(
    lines_tools: np.array,
    lines_objects,
    deduplicate_points_q: bool = False,
    eps: float = l_incidence_tol,
) -> float:
    # compute intersections

    output = [None for _ in range(lines_objects.shape[0])]
    for i, line_obj in enumerate(lines_objects):
        line_equality = [
            np.all(np.isclose(line_tool, line_obj)) for line_tool in lines_tools
        ]
        tuple_idx = np.where(line_equality)
        if len(tuple_idx) > 0:
            idx = tuple_idx[0]
            filtered_lines_tools = np.delete(lines_tools, idx, axis=0)
            output[i] = lines_line_intersection(
                filtered_lines_tools, line_obj[0], line_obj[1], eps
            )
        else:
            output[i] = lines_line_intersection(
                lines_tools, line_obj[0], line_obj[1], eps
            )

    if deduplicate_points_q:
        points = np.empty((0, 3), dtype=float)
        for points_in_line in output:
            for point in points_in_line:
                points = np.append(points, point, axis=0)
        # Define domains to be subtracted
        points_rounded = np.round(points, decimals=collapse_precision)
        _, idx = np.unique(points_rounded, axis=0, return_index=True)
        unique_points = points[idx]
        return unique_points
    else:
        return output


def line_line_plot(a: np.array, b: np.array, c: np.array, d: np.array):
    out = line_line_intersection(a, b, c, d)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # Plot line segments
    ax.plot3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], "o-", label="Segment a-b")
    ax.plot3D([c[0], d[0]], [c[1], d[1]], [c[2], d[2]], "o-", label="Segment c-d")

    if len(out) > 0:
        x, y, z = out[:, 0], out[:, 1], out[:, 2]
        if out.shape == (1, 3):
            ax.plot3D(x, y, z, "o-", label="Intersection point")
        else:
            ax.plot3D(
                [c[0], d[0]],
                [c[1], d[1]],
                [c[2], d[2]],
                "o-",
                label="Intersection segment",
            )

    # Set labels for axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def coplanar_measurements(
    line: np.array, a: np.array, b: np.array, c: np.array
) -> float:
    p, q = line
    p_t_equ = np.hstack((np.array([a, b, c, p]), np.ones((4, 1))))
    measurement_p = np.linalg.det(p_t_equ) / 6.0

    q_t_equ = np.hstack((np.array([a, b, c, q]), np.ones((4, 1))))
    measurement_q = np.linalg.det(q_t_equ) / 6.0
    return measurement_p, measurement_q


def line_triangle_incidence(
    line: np.array, a: np.array, b: np.array, c: np.array, eps: float = s_incidence_tol
) -> bool:
    measurement_p, measurement_q = coplanar_measurements(line, a, b, c)
    p_triangle_incidence_q = np.isclose(measurement_p, 0.0, rtol=eps, atol=eps)
    q_triangle_incidence_q = np.isclose(measurement_q, 0.0, rtol=eps, atol=eps)

    if p_triangle_incidence_q or q_triangle_incidence_q:
        return True
    elif (measurement_p < 0.0 or p_triangle_incidence_q) and (
        measurement_q > 0.0 or q_triangle_incidence_q
    ):
        return True
    elif (measurement_p > 0.0 or p_triangle_incidence_q) and (
        measurement_q < 0.0 or q_triangle_incidence_q
    ):
        return True
    else:
        return False


def line_triangle_intersection(
    line: np.array, a: np.array, b: np.array, c: np.array, eps: float = s_incidence_tol
) -> np.array:

    out = line_triangle_incidence(line, a, b, c, eps)
    if not out:
        return None

    p, q = line

    # point to triangle intersections
    p_intersection = point_triangle_intersection(p, a, b, c, eps)
    q_intersection = point_triangle_intersection(q, a, b, c, eps)
    if (p_intersection is not None) and (q_intersection is not None):
        return line

    # line to triangle boundary intersections
    lines = np.array([[a, b], [b, c], [c, a]])
    out = lines_line_intersection(lines, p, q, eps)
    if out.shape[0] != 0:
        out_rounded = np.round(out, decimals=collapse_precision)
        _, idx = np.unique(out_rounded, axis=0, return_index=True)
        out = out[idx]
        if out.shape[0] == 1:
            return out[0]
        else:
            return out[:,0,:]
    else:
        if p_intersection is not None:
            return np.array([p])
        if q_intersection is not None:
            return np.array([q])

    # line to triangle intersection
    # Line segment direction vector
    line_dir = q - p

    # Edge vectors
    ab = b - a
    ac = c - a

    # Calculate normal of the triangle
    normal = np.cross(ab, ac)
    denom = np.dot(normal, line_dir)
    if np.isclose(denom, 0.0, rtol=eps, atol=eps):
        return None

    # Calculate the distance from line_start to the plane of the triangle
    d = np.dot(normal, a - p) / denom
    if d < 0 or d > 1:
        return False

    # Calculate the intersection point on the line
    intx_point = p + d * line_dir
    intx_point = point_triangle_intersection(intx_point, a, b, c, eps)
    if intx_point is None:
        return None
    else:
        return np.array([intx_point])


def lines_triangle_intersection(
    lines: np.array, a: np.array, b: np.array, c: np.array, eps: float = s_incidence_tol
) -> np.array:

    # compute intersections
    lines_triangle_int = partial(line_triangle_intersection, a=a, b=b, c=c, eps=eps)
    result = list(map(lines_triangle_int, lines))

    def data_type(data):
        if isinstance(data, np.ndarray):
            return True
        else:
            return False

    intx_q = np.array([data_type(data) for data in result])

    # filter points outside segment
    try:
        result = np.array(list(filter(lambda x: x is not None, result)))
    except:
        result = np.concatenate(list(filter(lambda x: x is not None, result)))

    return result, intx_q


def __line_triangles_intersection(
    line: np.array, triangles: np.array, eps: float = s_incidence_tol
) -> float:

    result = []
    # compute intersections
    for triangle in triangles:
        a, b, c = triangle
        out = line_triangle_intersection(line, a, b, c, eps)
        result.append(out)

    return result


def line_polygon_intersection(
    line: np.array, polygon_points: np.array, eps: float = s_incidence_tol
) -> np.array:

    triangles = triangulate_polygon(polygon_points)
    # compute intersections
    results = __line_triangles_intersection(line, triangles, eps)

    def data_type(data):
        if isinstance(data, np.ndarray):
            return True
        else:
            return False

    triangle_idx_q = np.array([data_type(data) for data in results])
    if np.any(triangle_idx_q):
        intx_data = np.empty((0, 3), float)
        for i, out in enumerate(results):
            if triangle_idx_q[i]:
                if len(out.shape) > 2:
                    for intx_point in out:
                        intx_data = np.append(intx_data, intx_point, axis=0)
                else:
                    intx_data = np.append(intx_data, out, axis=0)

        # Define domains to be subtracted
        intx_data_rounded = np.round(intx_data, decimals=collapse_precision)
        _, idx = np.unique(intx_data_rounded, axis=0, return_index=True)
        intx_data = intx_data[idx]
        a, b = line
        idx_sort = points_line_argsort(intx_data, a, b)
        if intx_data.shape[0] > 2:  # return only external intersections
            return intx_data[idx_sort[np.array([0, -1])]]
        else:
            return intx_data[idx_sort]
    else:
        return None


def lines_polygon_intersection(
    lines: np.array, polygon_points: np.array, eps: float = s_incidence_tol
) -> float:
    # compute intersections
    line_line_int = partial(
        line_polygon_intersection, polygon_points=polygon_points, eps=eps
    )
    result = list(map(line_line_int, lines))

    def data_type(data):
        if isinstance(data, np.ndarray):
            return True
        else:
            return False

    intx_q = np.array([data_type(data) for data in result])

    # filter points outside segment
    result = np.array(list(filter(lambda x: x is not None, result)))
    return result, intx_q
