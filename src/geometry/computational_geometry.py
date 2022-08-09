import warnings
import numpy as np
import matplotlib.pyplot as plt
import math


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

    def line_plane_intersection(
        self, plane: np.array, p: np.array, q: np.array
    ) -> np.array:
        t_row = np.ones((1, 4))
        n_data = np.array([plane[0], plane[1], plane[2], p]).T
        d_data = np.array([plane[0], plane[1], plane[2], q - p]).T
        n_equ = np.vstack((np.array([[1.0, 1.0, 1.0, 1.0]]), n_data))
        d_equ = np.vstack((np.array([[1.0, 1.0, 1.0, 0.0]]), d_data))
        tv = -np.linalg.det(n_equ) / np.linalg.det(d_equ)
        t_intersection_q = 0.0 <= tv and tv <= 1.0
        if not t_intersection_q:
            warnings.warn(
                "line_plane_intersection:: intersection ocurrs outside segment."
            )
        p_intersection = p + tv * (q - p)
        return p_intersection

    def intersect_with_coplanar_point(
        self, t_triangle: np.array, p: np.array, drop: np.array
    ) -> (bool, np.array, np.array):
        result = (False, np.array, np.array)
        return result

    def intersect_with_coplanar_segment(
        self, t_triangle: np.array, s: np.array, drop: np.array
    ) -> (bool, np.array, np.array):
        result = (False, np.array, np.array)
        return result

    def intersect_with_non_coplanar_segments(
        self, t_triangle: np.array, s_1: np.array, s_2: np.array, drop: np.array
    ) -> (bool, np.array, np.array):
        result = (False, np.array, np.array)
        return result

    def trinagle_trinagle_intersection(
        self, o_triangle: np.array, t_triangle: np.array
    ) -> (bool, np.array, np.array):
        sign = lambda x: math.copysign(1, x)

        p1, q1, r1 = o_triangle
        p2, q2, r2 = t_triangle

        volp1 = self.coplanar_measurement(p2, q2, r2, p1)
        volq1 = self.coplanar_measurement(p2, q2, r2, q1)
        volr1 = self.coplanar_measurement(p2, q2, r2, r1)

        volp2 = self.coplanar_measurement(p1, q1, r1, p2)
        volq2 = self.coplanar_measurement(p1, q1, r1, q2)
        volr2 = self.coplanar_measurement(p1, q1, r1, r2)

        p1_coplanarity_q = np.abs(volp1) < self.eps
        q1_coplanarity_q = np.abs(volq1) < self.eps
        r1_coplanarity_q = np.abs(volr1) < self.eps

        p2_coplanarity_q = np.abs(volp2) < self.eps
        q2_coplanarity_q = np.abs(volq2) < self.eps
        r2_coplanarity_q = np.abs(volr2) < self.eps

        t1_coplanarity_q = p1_coplanarity_q and q1_coplanarity_q and r1_coplanarity_q
        t2_coplanarity_q = p2_coplanarity_q and q2_coplanarity_q and r2_coplanarity_q

        t1_intersection_q = sign(volp1) == sign(volq1) == sign(volr1)
        t2_intersection_q = sign(volp2) == sign(volq2) == sign(volr2)

        if t1_coplanarity_q or t2_coplanarity_q:
            print("T_o and T_t are coplanar.")

        if t1_intersection_q:
            print("T_o does not intersect the plane of T_t.")

        if t2_intersection_q:
            print("T_r does not intersect the plane of T_o.")

        if t1_intersection_q or t2_intersection_q:
            return (False, np.array, np.array)

        dir_cross = np.cross(p2 - q2, r2 - q2)
        n_t = dir_cross / np.linalg.norm(dir_cross)
        drop = np.argmax(np.abs(n_t))

        discard_pq_q = sign(volp1) == sign(volq1) != 0
        discard_qr_q = sign(volq1) == sign(volr1) != 0
        discard_rp_q = sign(volr1) == sign(volp1) != 0

        result = (False, np.array, np.array)

        non_coplanar_segments_q = discard_pq_q or discard_qr_q or discard_rp_q
        if non_coplanar_segments_q:
            s_1: np.array = None
            s_2: np.array = None

            if discard_pq_q:
                print("Intersection could lie on segments qr and/or rp")
                s_1 = np.array([q, r])
                s_2 = np.array([r, p])

            if discard_qr_q:
                print("Intersection could lie on segments pq and/or rp")
                s_1 = np.array([p, q])
                s_2 = np.array([r, p])

            if discard_rp_q:
                print("Intersection could lie on segments pq and/or qr")
                s_1 = np.array([p, q])
                s_2 = np.array([q, r])

            result = intersect_with_non_coplanar_segments(t_triangle, s_1, s_2, drop)
            return result

        pq_is_coplanar_q = p1_coplanarity_q and q1_coplanarity_q
        qr_is_coplanar_q = q1_coplanarity_q and r1_coplanarity_q
        rp_is_coplanar_q = r1_coplanarity_q and p1_coplanarity_q

        coplanar_segment_q = pq_is_coplanar_q or qr_is_coplanar_q or rp_is_coplanar_q

        if coplanar_segment_q:
            s: np.array = None
            if pq_is_coplanar_q:
                print("Segment pq is coplanar")
                s = np.array([p, q])

            if qr_is_coplanar_q:
                print("Segment qr is coplanar")
                s = np.array([q, r])

            if rp_is_coplanar_q:
                print("Segment rp is coplanar")
                s = np.array([r, p])

            result = intersect_with_non_coplanar_segments(t_triangle, s, drop)
            return result

        coplanar_point_q = p1_coplanarity_q or q1_coplanarity_q or r1_coplanarity_q

        if coplanar_point_q:
            point: np.array = None
            if p1_coplanarity_q:
                print("Point p is on t_trinagle plane")
                point = p

            if q1_coplanarity_q:
                print("Point q is on t_trinagle plane")
                point = q

            if r1_coplanarity_q:
                print("Point r is on t_trinagle plane")
                point = r

            result = intersect_with_coplanar_point(t_triangle, s, drop)
            return result

        return result
