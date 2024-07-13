import numpy as np
from functools import partial
from globals import geometry_collapse_tol as collapse_tol
from globals import geometry_point_line_incidence_tol as l_incidence_tol
from globals import geometry_point_polygon_incidence_tol as incidence_tol
from geometry.operations.polygon_geometry_operations import triangulate_polygon



def colinear_measurement(p: np.array, a: np.array, b: np.array) -> float:
    v = a - p
    u = b - p
    ve, ue = np.array([v, u])
    measurement = 0.5 * np.linalg.norm(np.cross(ve, ue))
    return measurement

def point_line_incidence(
    p: np.array, a: np.array, b: np.array, eps: float = l_incidence_tol
) -> float:
    measurement = colinear_measurement(a, b, p)
    point_line_incidence_q = np.isclose(measurement, 0.0, rtol=eps, atol=eps)
    return point_line_incidence_q


def point_line_intersection(
    p: np.array, a: np.array, b: np.array, eps: float = l_incidence_tol
) -> float:

    point_a_incidence_q = np.all(np.isclose(a, p, rtol=collapse_tol, atol=collapse_tol))
    if point_a_incidence_q:
        return a

    point_b_incidence_q = np.all(np.isclose(b, p, rtol=collapse_tol, atol=collapse_tol))
    if point_b_incidence_q:
        return b

    point_line_incidence_q = point_line_incidence(a, b, p, eps)
    if point_line_incidence_q:

        # sphere centered in a
        bta_norm = np.linalg.norm(b - a)
        pta_norm = np.linalg.norm(p - a)

        # sphere centered in b
        atb_norm = np.linalg.norm(a - b)
        ptb_norm = np.linalg.norm(p - b)

        # sphere centered in a
        a_predicate = bta_norm > pta_norm or np.isclose(
            bta_norm, pta_norm, rtol=eps, atol=eps
        )

        # sphere centered in b
        b_predicate = atb_norm > ptb_norm or np.isclose(
            atb_norm, ptb_norm, rtol=eps, atol=eps
        )
        if a_predicate and b_predicate:
            return p
        else:
            return None
    else:
        return None


def points_line_intersection(
    points: np.array, a: np.array, b: np.array, eps: float = l_incidence_tol
) -> float:
    # compute intersections
    point_line_int = partial(point_line_intersection, a=a, b=b, eps=eps)
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


def points_line_argsort(
    points: np.array, a: np.array, b: np.array, ba_sorting: bool = False
) -> float:
    # by default assuming segment is oriented from a to b
    if ba_sorting:
        idx = np.argsort(np.linalg.norm(points - b, axis=1))
    else:
        idx = np.argsort(np.linalg.norm(points - a, axis=1))
    return idx

def coplanar_measurement(p: np.array, a: np.array, b: np.array, c: np.array) -> float:
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
    if (u >= 0) and (v >= 0) and (u + v <= 1):
        return p
    else:
        return None


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


def __point_triangles_intersection(
    p: np.array, triangles: np.array, eps: float = incidence_tol
) -> float:

    result = []
    # compute intersections
    for triangle in triangles:
        a, b, c = triangle
        out = point_triangle_intersection(p, a, b, c, eps)
        result.append(out)

    return result


def point_polygon_intersection(
    p: np.array, polygon_points: np.array, eps: float = incidence_tol
) -> float:

    triangles = triangulate_polygon(polygon_points)
    # compute intersections
    results = __point_triangles_intersection(p, triangles, eps)

    def data_type(data):
        if isinstance(data, np.ndarray):
            return True
        else:
            return False

    triangle_idx_q = np.array([data_type(data) for data in results])
    if np.any(triangle_idx_q):
        return p
    else:
        return None


def points_polygon_intersection(
    points: np.array, polygon_points: np.array, eps: float = incidence_tol
) -> float:
    # compute intersections
    point_line_int = partial(
        point_polygon_intersection, polygon_points=polygon_points, eps=eps
    )
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
