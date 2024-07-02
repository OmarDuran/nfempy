from functools import partial
from globals import topology_point_line_incidence_tol as p_incidence_tol
from globals import topology_line_line_incidence_tol as l_incidence_tol

import numpy as np
import matplotlib.pyplot as plt
from topology.point_line_incidence import point_line_intersection
from topology.point_line_incidence import points_line_orientation


def line_line_intersection(
    a: np.array, b: np.array, c: np.array, d: np.array, eps: float = l_incidence_tol
) -> float:

    out = points_line_orientation(np.array([a, b]), c, d, p_incidence_tol)
    if len(out) > 0:
        return out

    # Direction vectors
    d1 = b - a
    d2 = d - c

    # Matrix form Ax = b
    A = np.array([d1, -d2]).T
    b = c - a

    # Solve the least squares problem
    t, s = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the intersection points
    pa = a + t * d1
    pc = c + s * d2
    # Check if the intersection points are the same
    if np.allclose(pa, pc, rtol=eps, atol=eps):
        out = point_line_intersection(pa, c, d)
        if out is not None:
            return np.array([pa])
        else:
            return None
    else:
        return None  # No intersection


def lines_line_intersection(
    lines: np.array, a: np.array, b: np.array, eps: float = l_incidence_tol
) -> float:
    # compute intersections
    line_line_int = partial(line_line_intersection, c=a, d=b, eps=eps)
    result = [line_line_int(line[0], line[1]) for line in lines]

    # filter points outside segment
    result = np.array(list(filter(lambda x: x is not None, result)))
    return result


def lines_lines_intersection(
    lines_tools: np.array, lines_objects, eps: float = l_incidence_tol
) -> float:
    # compute intersections

    result = lines_objects.shape[0] * [None]
    for i, line_obj in enumerate(lines_objects):
        line_equality = [
            np.all(np.isclose(line_tool, line_obj)) for line_tool in lines_tools
        ]
        tuple_idx = np.where(line_equality)
        if len(tuple_idx) > 0:
            idx = tuple_idx[0]
            filtered_lines_tools = np.delete(lines_tools, idx, axis=0)
            result[i] = lines_line_intersection(
                filtered_lines_tools, line_obj[0], line_obj[1], eps
            )
        else:
            result[i] = lines_line_intersection(
                lines_tools, line_obj[0], line_obj[1], eps
            )
    return result


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
