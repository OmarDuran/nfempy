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

    def render_lines(
        self, a: np.array, b: np.array, p: np.array, q: np.array, pos: np.array
    ):
        ar, br, pr, qr = np.array([a, b, p, q])[:, pos]
        lab = np.array([ar, br])
        lpq = np.array([pr, qr])
        plt.plot(lab[:, 0], lab[:, 1], color="red", marker="o", lw=2)
        plt.plot(lpq[:, 0], lpq[:, 1], color="blue", marker="o", lw=2)
        plt.show()

    def line_line_intersection(
        self,
        a: np.array,
        b: np.array,
        p: np.array,
        q: np.array,
        pos: np.array,
        render_lines_q: bool = False,
    ) -> (bool, np.array, np.array):
        ar, br, pr, qr = np.array([a, b, p, q])[:, pos]
        if render_lines_q:
            self.render_lines(a, b, p, q, pos)
        dv = np.linalg.det(np.array([ar - br, pr - qr]).T)
        parallel_lines_q = np.abs(dv) < self.eps
        if parallel_lines_q:
            p_is_ab_colinear_q = (
                np.abs(self.colinear_measurement(a, b, p, pos)) < self.eps
            )
            q_is_ab_colinear_q = (
                np.abs(self.colinear_measurement(a, b, q, pos)) < self.eps
            )
            if not p_is_ab_colinear_q and not q_is_ab_colinear_q:
                return (False, np.array, np.array)
            tau = (ar - br) / np.linalg.norm(ar - br)
            n = np.array([tau[1], -tau[1]])

            drop = np.argmax(np.abs(n))
            pos = np.array(
                list(
                    {
                        0,
                        1,
                    }
                    - {drop}
                )
            )

            ar_member_q = pr[pos] <= ar[pos] and ar[pos] <= qr[pos]
            br_member_q = pr[pos] <= br[pos] and br[pos] <= qr[pos]
            if ar_member_q or br_member_q:
                if ar_member_q and br_member_q:
                    return (True, a, b)
                if ar_member_q:
                    return (True, a, q)
                if br_member_q:
                    return (True, p, b)
            else:
                return (True, p, q)

        tv = np.linalg.det(np.array([ar - pr, pr - qr]).T) / dv
        uv = np.linalg.det(np.array([ar - pr, ar - br]).T) / dv
        t_intersection_q = 0.0 <= tv and tv <= 1.0
        u_intersection_q = 0.0 <= uv and uv <= 1.0
        p_intersection = None
        if t_intersection_q and u_intersection_q:
            p_intersection = a + tv * (b - a)
            return (True, p_intersection, np.array)
        else:
            return (False, np.array, np.array)
