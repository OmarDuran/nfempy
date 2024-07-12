from functools import partial
from globals import topology_point_polygon_incidence_tol as incidence_tol
from globals import topology_collapse_tol as collapse_tol
from topology.point_line_incidence import point_line_intersection
import numpy as np


def coplanar_measurement(
    p: np.array, a: np.array, b: np.array, c: np.array
) -> float:
    t_equ = np.hstack((np.array([a, b, c, p]), np.ones((4, 1))))
    measurement = np.linalg.det(t_equ) / 6.0
    return measurement


def point_triangle_incidence(
    p: np.array, a: np.array, b: np.array, c: np.array, eps: float = incidence_tol
) -> float:
    measurement = coplanar_measurement(p, a, b, c)
    point_triangle_incidence_q = np.isclose(measurement, 0.0, rtol=eps, atol=eps)
    return point_triangle_incidence_q


def point_triangle_intersection(
    p: np.array, a: np.array, b: np.array, c: np.array, eps: float = incidence_tol
) -> np.array:

    point_a_incidence_q = np.all(np.isclose(a, p, rtol=collapse_tol, atol=collapse_tol))
    if point_a_incidence_q:
        return a

    point_b_incidence_q = np.all(np.isclose(b, p, rtol=collapse_tol, atol=collapse_tol))
    if point_b_incidence_q:
        return b

    point_c_incidence_q = np.all(np.isclose(c, p, rtol=collapse_tol, atol=collapse_tol))
    if point_c_incidence_q:
        return c

    point_triangle_incidence_q = point_triangle_incidence(p, a, b, c, eps)
    if not point_triangle_incidence_q:
        return None

    point_line_ab_out = point_line_intersection(p, a, b, eps)
    if point_line_ab_out is not None:
        return point_line_ab_out
    point_line_bc_out = point_line_intersection(p, b, c, eps)
    if point_line_bc_out is not None:
        return point_line_bc_out
    point_line_ca_out = point_line_intersection(p, c, a, eps)
    if point_line_ca_out is not None:
        return point_line_ca_out

    # Vectors from a to b, a to c, and a to p
    ab = b - a
    ac = c - a
    ap = p - a

    # Compute dot products
    dot_ab_ab = np.dot(ab, ab)
    dot_ab_ac = np.dot(ab, ac)
    dot_ac_ac = np.dot(ac, ac)
    dot_ap_ab = np.dot(ap, ab)
    dot_ap_ac = np.dot(ap, ac)

    # Compute barycentric coordinates
    denom = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac
    u = (dot_ac_ac * dot_ap_ab - dot_ab_ac * dot_ap_ac) / denom
    v = (dot_ab_ab * dot_ap_ac - dot_ab_ac * dot_ap_ab) / denom

    # Check if point is inside the triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def points_triangle_intersection(
    points: np.array, a: np.array, b: np.array, c: np.array, eps: float = incidence_tol
) -> float:
    # compute intersections
    point_line_int = partial(point_triangle_intersection, a=a, b=b, c=c, eps=eps)
    result = list(map(point_line_int, points))

    def data_type(data):
        if isinstance(data, np.ndarray):
            return True
        else:
            return False

    intx_q = np.array([data_type(data) for data in result])

    # filter points outside segment
    result = np.array(list(filter(lambda x: x is not None, result)))
    return result, intx_q

