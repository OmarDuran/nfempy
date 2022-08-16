import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import geometry.triangle_triangle_intersection_test as tt_intersector


class PolygonPolygonIntersectionTest:
    """Worker class for polygon-polygon intersection in R3."""

    def __init__(self, eps: float = 1.0e-12):
        self.eps = eps
        self.tt_intersector = tt_intersector.TriangleTriangleIntersectionTest(eps)
        self.oc = np.array([[0, 1, 3], [1, 2, 3]])
        self.tc = np.array([[0, 1, 3], [1, 2, 3]])
        self.poly_line: np.array = None

    def render_polygons(self, o_polygon: np.array, t_polygon: np.array):
        axes = plt.axes(projection="3d")
        axes.set_xlabel("$X$", fontsize=12)
        axes.set_ylabel("$Y$", fontsize=12)
        axes.set_zlabel("$Z$", fontsize=12)

        ox = o_polygon[[0, 1, 2, 3, 0]].T[0]
        oy = o_polygon[[0, 1, 2, 3, 0]].T[1]
        oz = o_polygon[[0, 1, 2, 3, 0]].T[2]
        axes.plot(ox, oy, oz, color="red", marker="o", lw=2)

        tx = t_polygon[[0, 1, 2, 3, 0]].T[0]
        ty = t_polygon[[0, 1, 2, 3, 0]].T[1]
        tz = t_polygon[[0, 1, 2, 3, 0]].T[2]
        axes.plot(tx, ty, tz, color="blue", marker="o", lw=2)

        if self.poly_line is not None:
            n, m = self.poly_line.shape
            r = [i for i in range(n)]
            r.append(0)
            lx = self.poly_line[r].T[0]
            ly = self.poly_line[r].T[1]
            lz = self.poly_line[r].T[2]
            axes.plot(lx, ly, lz, color="black", marker="o", lw=2)

        plt.show()

    def reduce_poly_line(self, polygon: np.array):
        triangle = polygon[self.tc[0]]
        p, q, r = triangle
        dir_cross = np.cross(p - q, r - q)
        n_t = dir_cross / np.linalg.norm(dir_cross)
        drop = np.argmax(np.abs(n_t))
        pos = np.array(
            list(
                {
                    0,
                    1,
                    2,
                }
                - {drop}
            )
        )
        poly_line_r = self.poly_line[:, pos]
        if poly_line_r.shape[0] > 1:
            ar, br = poly_line_r[[0, 1]]
            tau = (ar - br) / np.linalg.norm(ar - br)
            n = np.array([tau[1], -tau[0]])
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

            poly_line_r_1d = poly_line_r[:, pos]
            begin_id = np.argmin(poly_line_r_1d)
            end_id = np.argmax(poly_line_r_1d)
            self.poly_line = self.poly_line[[begin_id, end_id]]

    # def build_connectivity(self, connectivity, polygon):
    #     m, n = polygon.shape
    #     xc = np.mean(a, axis=0)
    #     connectivity

    def polygon_polygon_intersection(
        self, o_polygon: np.array, t_polygon: np.array, render_polygons_q: bool = False
    ) -> (bool, np.array, np.array):

        result = (False, np.array, np.array)
        opoly = [o_polygon[self.oc[0]], o_polygon[self.oc[1]]]
        tpoly = [t_polygon[self.tc[0]], t_polygon[self.tc[1]]]

        intersection_data = []
        for ot in opoly:
            for tt in tpoly:
                intersection = self.tt_intersector.triangle_triangle_intersection(
                    ot, tt
                )
                intersection_data.append(intersection)

        i_results = [chunk[0] for chunk in intersection_data]
        n_intersections = np.count_nonzero(i_results)

        if n_intersections != 0:
            points = []
            for chunk in intersection_data:
                if chunk[0]:
                    points.append(chunk[1])
                    points.append(chunk[2])
            self.poly_line = np.array(points)
            self.reduce_poly_line(o_polygon)
            result = (True, self.poly_line[0], self.poly_line[1])

        if render_polygons_q:
            self.render_polygons(o_polygon, t_polygon)

        return result
