from functools import partial
from globals import topology_point_line_incidence_tol as incidence_tol
from globals import topology_collapse_tol as collapse_tol
import numpy as np

def colinear_measurement(p: np.array, a: np.array, b: np.array) -> float:
    v = a - p
    u = b - p
    ve, ue = np.array([v, u])
    measurement = 0.5 * np.linalg.norm(np.cross(ve, ue))
    return measurement

def point_line_incidence(p: np.array, a: np.array, b: np.array, eps: float = incidence_tol) -> float:
    measurement = colinear_measurement(a,b,p)
    point_line_incidence_q = np.isclose(measurement,0.0, rtol=eps, atol=eps)
    return point_line_incidence_q

def point_line_intersection(p: np.array, a: np.array, b: np.array, eps: float = incidence_tol) -> float:

    point_a_incidence_q = np.all(np.isclose(a, p, rtol=collapse_tol, atol=collapse_tol))
    if point_a_incidence_q:
        return a

    point_b_incidence_q = np.all(np.isclose(b, p, rtol=collapse_tol, atol=collapse_tol))
    if point_b_incidence_q:
        return b

    point_line_incidence_q = point_line_incidence(a,b,p,eps)
    if point_line_incidence_q:

        # sphere centered in a
        bta_norm = np.linalg.norm(b - a)
        pta_norm = np.linalg.norm(p - a)

        # sphere centered in b
        atb_norm = np.linalg.norm(a - b)
        ptb_norm = np.linalg.norm(p - b)

        # sphere centered in a
        a_predicate = bta_norm > pta_norm or np.isclose(bta_norm, pta_norm, rtol=eps, atol=eps)

        # sphere centered in b
        b_predicate = atb_norm > ptb_norm or np.isclose(atb_norm, ptb_norm, rtol=eps, atol=eps)
        if a_predicate and b_predicate:
            return p
        else:
            return None
    else:
        return None

def points_line_intersection(points: np.array, a: np.array, b: np.array, eps: float = incidence_tol) -> float:
    # compute intersections
    point_line_int = partial(point_line_intersection,a=a,b=b,eps=eps)
    result = list(map(point_line_int,points))

    # filter points outside segment
    result = np.array(list(filter(lambda x: x is not None, result)))
    return result

def points_line_orientation(points: np.array, a: np.array, b: np.array, eps: float = incidence_tol) -> float:
    points = np.array(points_line_intersection(points, a, b, eps))
    if points.shape[0] == 0:
        return points
    # assuming segment is oriented from a to b
    idx = np.argsort(np.linalg.norm(points - a, axis=1))
    return points[idx]