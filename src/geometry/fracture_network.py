import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import linalg as la

import geometry.polygon_polygon_intersection_test as pp_intersector
import geometry.triangle_triangle_intersection_test as tt_intersector

from .cell import Cell


class FractureNetwork:

    # The network object represents a mesh of intersecting objects.
    # For 2d the mesh includes (d-f)-facets f = {0,1}:
    #   (0)-facets vertices
    #   (1)-facets edges

    # For 3d the mesh includes (d-f)-facets f = {0,1,2}:
    #   (0)-facets vertices
    #   (1)-facets edges
    #   (2)-facets faces

    def __init__(self, dimension, eps: float = 1.0e-12):

        self.eps = eps
        self.cells = np.array([], dtype=Cell)
        self.points: np.array = None
        self.vertices = None
        self.dimension = dimension
        self.grahp = None
        self.connectivity = None
        self.fracture_tags = None
        self.physical_tag_shift = 1

    def render_fractures(self, fractures: np.array):
        axes = plt.axes(projection="3d")
        axes.set_xlabel("$X$", fontsize=12)
        axes.set_ylabel("$Y$", fontsize=12)
        axes.set_zlabel("$Z$", fontsize=12)

        for fracture in fractures:
            ox = fracture[[0, 1, 2, 3, 0]].T[0]
            oy = fracture[[0, 1, 2, 3, 0]].T[1]
            oz = fracture[[0, 1, 2, 3, 0]].T[2]
            axes.plot(ox, oy, oz, color="green", marker="o", lw=2)

        plt.show()

    def insert_vertex_cell(self, point, physical_tag):
        R = self.points - point
        r_norm = la.norm(R, axis=1)
        index = np.where(r_norm < self.eps)
        vertex = None
        if index[0].size == 0:
            point_id = len(self.points)
            self.points = np.append(self.points, [point], axis=0)
            cell_id = len(self.cells)
            vertex = Cell(0, cell_id, point_id=point_id, physical_tag=physical_tag)
            self.cells = np.append(self.cells, vertex)
        else:
            target_point_id = index[0][0]
            for cell_i in self.cells:
                if cell_i.dimension == 0 and cell_i.point_id == target_point_id:
                    vertex = cell_i
                    break
        return vertex

    def insert_fracture_cell(self, cell_id, fracture, physical_tag):

        self.points = np.append(
            self.points, np.array([point for point in fracture]), axis=0
        )
        loop = [i + cell_id for i in range(len(fracture))]
        self.cells = np.append(self.cells, np.array([Cell(0, index) for index in loop]))

        loop.append(loop[0])
        connectivities = np.array(
            [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
        )

        cell_id = cell_id + len(fracture)
        edges_indices = []
        for con in connectivities:
            edge = Cell(1, cell_id, physical_tag=physical_tag)
            edge.boundary_cells = self.cells[con]
            self.cells = np.append(self.cells, edge)
            edges_indices.append(cell_id)
            cell_id = cell_id + 1

        edges_indices = np.array(edges_indices)
        surface = Cell(2, cell_id)
        surface.boundary_cells = self.cells[edges_indices]
        self.cells = np.append(self.cells, surface)

        return cell_id

    def intersect_2D_fractures(self, fractures, render_intersection_q=False):

        self.cells = np.array([], dtype=Cell)
        self.points = np.empty((0, 3), dtype=float)

        cell_id = 0
        n_fractures = len(fractures)
        for fracture in fractures:
            cell_id = self.insert_fracture_cell(cell_id, fracture)
            cell_id = cell_id + 1

        intersection_data = [[None for f in fractures] for f in fractures]

        cells_2d = [cell_i for cell_i in self.cells if cell_i.dimension == 2]
        for i, cell_i in enumerate(cells_2d):
            f_i = fractures[i]
            for j, cell_j in enumerate(cells_2d):
                f_j = fractures[j]
                if i >= j:
                    intersection_data[i][j] = (False, np.array, np.array)
                    continue

                obj = pp_intersector.PolygonPolygonIntersectionTest()
                intersection_q = obj.polygon_polygon_intersection(
                    f_i, f_j, render_intersection_q
                )
                intersection_data[i][j] = intersection_q

        for i, cell_i in enumerate(cells_2d):

            i_results = [chunk[0] for chunk in intersection_data[i]]
            n_intersections = np.count_nonzero(i_results)

            if n_intersections > 0:

                p, q, r = fractures[i][[0, 1, 3]]
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

                fractures_1d = []
                for chunk in intersection_data[i]:
                    if chunk[0]:
                        fractures_1d.append(np.array([chunk[1], chunk[2]]))

                fracture_network = FractureNetwork(dimension=2)
                fracture_network.intersect_1D_fractures(
                    fractures_1d, pos, render_intersection_q
                )
                # fracture_network.build_grahp()
                # fracture_network.draw_grahp()

                # insert cells
                point_id = len(self.points)
                cells_id = len(self.cells)
                self.points = np.append(
                    self.points,
                    np.array([point for point in fracture_network.points]),
                    axis=0,
                )

                for f_cell in fracture_network.cells:
                    f_cell.id = f_cell.id + cells_id
                    if f_cell.dimension == 0:
                        f_cell.point_id = f_cell.point_id + point_id
                    self.cells = np.append(self.cells, f_cell)

                f_cell_1d = []
                if n_intersections == 1:
                    f_cell_1d = [
                        f_cell
                        for f_cell in fracture_network.cells
                        if f_cell.dimension == 1
                    ]
                else:
                    f_cell_1d = [
                        f_cell
                        for f_cell in fracture_network.cells
                        if f_cell.dimension == 1 and len(f_cell.immersed_cells) > 0
                    ]

                cj = 0
                for j, chunk in enumerate(intersection_data[i]):
                    if chunk[0]:
                        f_cell_intersection = f_cell_1d[cj]
                        cell_j = cells_2d[j]
                        cell_i.immersed_cells = np.append(
                            cell_i.immersed_cells, f_cell_intersection
                        )
                        cell_j.immersed_cells = np.append(
                            cell_j.immersed_cells, f_cell_intersection
                        )
                        cj = cj + 1

    def intersect_1D_fractures(
        self, fractures, pos=np.array([0, 1]), render_intersection_q=False
    ):

        self.fracture_tags = [
            i + self.physical_tag_shift for i, _ in enumerate(fractures)
        ]
        self.points = np.array([])
        self.cells = np.array([], dtype=Cell)

        # build dfn disjoint geometrical description
        points = []
        cell_id = 0
        point_id = 0
        for f_i, fracture in enumerate(fractures):
            physical_tag = self.fracture_tags[f_i]
            bc_cells = np.array([], dtype=Cell)
            for point in fracture:
                vertex = Cell(0, cell_id, point_id=point_id, physical_tag=physical_tag)
                self.cells = np.append(self.cells, np.array(vertex))
                point_id = point_id + 1
                points.append(point)
                bc_cells = np.append(bc_cells, vertex)
                cell_id = cell_id + 1

            edge = Cell(1, cell_id, physical_tag=physical_tag)
            edge.boundary_cells = bc_cells
            self.cells = np.append(self.cells, edge)
            cell_id = cell_id + 1
        self.points = np.stack(points, axis=0)

        tt_test = tt_intersector.TriangleTriangleIntersectionTest()

        cells_1d = [cell_1d for cell_1d in self.cells if cell_1d.dimension == 1]
        for i, cell_i in enumerate(cells_1d):
            cell_i_bc_ids = [bc_cell.point_id for bc_cell in cell_i.boundary_cells]
            a, b = [self.points[id] for id in cell_i_bc_ids]
            for j, cell_j in enumerate(cells_1d):
                cell_j_bc_ids = [bc_cell.point_id for bc_cell in cell_j.boundary_cells]
                p, q = [self.points[id] for id in cell_j_bc_ids]
                if i >= j:
                    continue
                intersection_data = tt_test.line_line_intersection(
                    a, b, p, q, pos, render_intersection_q
                )
                if intersection_data[0]:
                    point = intersection_data[1]
                    physical_tag = self.pair_intergers(
                        cell_i.physical_tag, cell_j.physical_tag
                    )
                    vertex = self.insert_vertex_cell(point, physical_tag)
                    cell_i.immersed_cells = np.append(cell_i.immersed_cells, vertex)
                    cell_j.immersed_cells = np.append(cell_j.immersed_cells, vertex)

        cell_id = len(self.cells)
        for cell_i in cells_1d:
            b, e = [bc_cell.id for bc_cell in cell_i.boundary_cells]
            i = [immersed_cell.point_id for immersed_cell in cell_i.immersed_cells]
            if len(i) > 0:
                R = self.points[i] - self.points[self.cells[b].point_id]
                r_norm = la.norm(R, axis=1)
                perm = np.argsort(r_norm)
                cell_indices = [cell_i.immersed_cells[k].id for k in perm.tolist()]
                cell_indices = list(dict.fromkeys(cell_indices))
                cell_indices.insert(0, b)
                cell_indices.append(e)
                connectivities = np.array(
                    [
                        [cell_indices[index], cell_indices[index + 1]]
                        for index in range(len(cell_indices) - 1)
                    ]
                )
                cell_i.immersed_cells = np.array([], dtype=Cell)
                for con in connectivities:
                    edge = Cell(1, cell_id, physical_tag=cell_i.physical_tag)
                    edge.boundary_cells = self.cells[con]
                    self.cells = np.append(self.cells, edge)
                    cell_i.immersed_cells = np.append(cell_i.immersed_cells, edge)
                    cell_id = cell_id + 1

    def pair_intergers(self, x, y):
        # http://szudzik.com/ElegantPairing.pdf
        z = None
        if max(x, y) is not x:
            z = y**2 + x
        else:
            z = x**2 + x + y
        return z

    def unpair_intergers(self, z):
        # http://szudzik.com/ElegantPairing.pdf
        pair = None
        test = z - np.sqrt(z) ** 2 < np.sqrt(z)
        if test:
            pair = [z - np.sqrt(z) ** 2, np.sqrt(z)]
        else:
            pair = [np.sqrt(z), z - np.sqrt(z) ** 2 - np.sqrt(z)]
        return pair

    def shift_point_ids(self, shift=0):
        cells_0d = [cell for cell in self.cells if cell.dimension == 0]
        for cell in cells_0d:
            cell.point_id = cell.point_id + shift

    def shift_cell_ids(self, shift=0):
        for cell in self.cells:
            cell.id = cell.id + shift

    def gather_graph_edges(self, g_cell: Cell, tuple_id_list):
        for bc_cell in g_cell.boundary_cells:
            tuple_id_list.append((g_cell.id, bc_cell.id))
            if bc_cell.dimension != 0:
                self.gather_graph_edges(bc_cell, tuple_id_list)

        for immersed_cell in g_cell.immersed_cells:
            tuple_id_list.append((g_cell.id, immersed_cell.id))
            if immersed_cell.dimension != 0:
                self.gather_graph_edges(immersed_cell, tuple_id_list)

    def build_grahp(self, all_fixed_d_cells_q=False):

        disjoint_cells = []
        if all_fixed_d_cells_q:
            disjoint_cells = [
                cell_i
                for cell_i in self.cells
                if cell_i.dimension == self.dimension - 1
            ]
        else:
            disjoint_cells = [
                cell_i
                for cell_i in self.cells
                if cell_i.dimension == self.dimension - 1
                and len(cell_i.immersed_cells) == 0
            ]

        tuple_id_list = []
        for cell_1d in disjoint_cells:
            self.gather_graph_edges(cell_1d, tuple_id_list)

        self.graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)

    def draw_grahp(self):
        nx.draw(
            self.graph,
            pos=nx.circular_layout(self.graph),
            with_labels=True,
            node_color="skyblue",
        )
