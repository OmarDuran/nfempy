import numpy as np
import matplotlib.pyplot as plt

class ComputationalGeometry:
    """Worker class for computational geometry.

    This class is mainly dedicated to compute polygon-polygon intersections

    """

    def __init__(self, eps: float = 1.0e-18):
        self.eps = eps

    def coplanar_measurement(
        self, a: np.array, b: np.array, c: np.array, p: np.array
    ) -> float:
        tequ = np.hstack((np.array([a, b, c, p]), np.ones((4, 1))))
        measurement = np.linalg.det(tequ) / 6.0
        return measurement

    def colinear_measurement(
        self, a: np.array, b: np.array, p: np.array, pos: np.array
    ) -> float:
        v = a - p
        u = b - p
        ve, ue = np.array([v, u])[:, pos]
        ve = np.append(ve, 0)
        ue = np.append(ue, 0)
        measurement = 0.5 * np.linalg.norm(np.cross(ve, ue))
        return measurement

    def point_in_triangle(
        self, a: np.array, b: np.array, c: np.array, p: np.array, pos: np.array
    ) -> bool:
        r_distances = [np.linalg.norm(x - p) for x in [a, b, c]]
        coincide_q = np.any(np.absolute(r_distances) < self.eps)
        if coincide_q:
            return True
        colinear_ms = [
            self.colinear_measurement(a, b, p, pos),
            self.colinear_measurement(b, c, p, pos),
            self.colinear_measurement(c, a, p, pos),
        ]
        colinearity_q = np.any(np.absolute(colinear_ms) < self.eps)
        if colinearity_q:
            return True
        ar, br, cr, pr = np.array([a, b, c, p])[:, pos]
        v0 = cr - ar
        v1 = br - ar
        v2 = pr - ar
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        dv = dot00 * dot11 - dot01 * dot01
        u = (dot11 * dot02 - dot01 * dot12) / dv
        v = (dot00 * dot12 - dot01 * dot02) / dv
        region_member_q = (u >= 0) and (v >= 0) and (u + v < 1)
        return region_member_q

    def line_line_intersection(self, a: np.array, b: np.array, p: np.array, q: np.array, pos: np.array, render_lines_q: bool = False) -> np.array:
        ar, br, pr, qr = np.array([a, b, p, q])[:, pos]
        if render_lines_q:
