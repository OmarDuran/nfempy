import networkx as nx
import numpy as np

from geometry.cell import Cell


class GeometryBuilder:
    def __init__(self, dimension):
        self.dimension = dimension
        self.cells = np.array([], dtype=Cell)
        self.points = np.empty((0, dimension), dtype=float)

    def set_boundary(self, vertices, connectivity, material_id):
        self.vertices = vertices
        self.connectivity = connectivity
        self.material_id = material_id

    def build_internal_bc(self, Network, normal_expansion=1.0e-1):
        assert (
            Network.dimension == self.dimension
        ), f"Geometry and network dimension are not equal {Network.dimension}"

        # classify intersections
        nodes = list(Network.grahp.nodes)
        node_neighs = [[] for _ in nodes]
        for i in range(len(nodes)):
            neighs = list(nx.all_neighbors(Network.grahp, nodes[i]))
            node_neighs[i].append(neighs)

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
            disjoint_cells = [cell_i for cell_i in self.cells]
        else:
            disjoint_cells = [
                cell_i for cell_i in self.cells if len(cell_i.immersed_cells) == 0
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

    def build_box_2D(self, box_points):

        self.points = np.append(
            self.points, np.array([point for point in box_points]), axis=0
        )
        loop = [i for i in range(len(box_points))]
        self.cells = np.append(self.cells, np.array([Cell(0, index) for index in loop]))

        loop.append(loop[0])
        connectivities = np.array(
            [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
        )

        cell_id = len(box_points)
        edges_indices = []
        for con in connectivities:
            edge = Cell(1, cell_id)
            edge.boundary_cells = self.cells[con]
            self.cells = np.append(self.cells, edge)
            edges_indices.append(cell_id)
            cell_id = cell_id + 1

        edges_indices = np.array(edges_indices)
        surface = Cell(2, cell_id)
        surface.boundary_cells = self.cells[edges_indices]
        self.cells = np.append(self.cells, surface)

    def build_box(cells, box_points):

        cells = np.append(
            cells, np.array([cell(0, i) for i, point in enumerate(box_points)])
        )

        edge = cell(1, 8)
        edge.boundary_cells = cells[[0, 1]]
        cells = np.append(cells, edge)

        edge = cell(1, 9)
        edge.boundary_cells = cells[[1, 2]]
        cells = np.append(cells, edge)

        edge = cell(1, 10)
        edge.boundary_cells = cells[[2, 3]]
        cells = np.append(cells, edge)

        edge = cell(1, 11)
        edge.boundary_cells = cells[[3, 0]]
        cells = np.append(cells, edge)

        edge = cell(1, 12)
        edge.boundary_cells = cells[[4, 5]]
        cells = np.append(cells, edge)

        edge = cell(1, 13)
        edge.boundary_cells = cells[[5, 6]]
        cells = np.append(cells, edge)

        edge = cell(1, 14)
        edge.boundary_cells = cells[[6, 7]]
        cells = np.append(cells, edge)

        edge = cell(1, 15)
        edge.boundary_cells = cells[[7, 4]]
        cells = np.append(cells, edge)

        edge = cell(1, 16)
        edge.boundary_cells = cells[[0, 4]]
        cells = np.append(cells, edge)

        edge = cell(1, 17)
        edge.boundary_cells = cells[[1, 5]]
        cells = np.append(cells, edge)

        edge = cell(1, 18)
        edge.boundary_cells = cells[[2, 6]]
        cells = np.append(cells, edge)

        edge = cell(1, 19)
        edge.boundary_cells = cells[[3, 7]]
        cells = np.append(cells, edge)

        surface = cell(2, 20)
        surface.boundary_cells = cells[[8, 17, 12, 16]]
        cells = np.append(cells, surface)

        surface = cell(2, 21)
        surface.boundary_cells = cells[[9, 18, 13, 17]]
        cells = np.append(cells, surface)

        surface = cell(2, 22)
        surface.boundary_cells = cells[[10, 13, 14, 19]]
        cells = np.append(cells, surface)

        surface = cell(2, 23)
        surface.boundary_cells = cells[[11, 16, 15, 19]]
        cells = np.append(cells, surface)

        surface = cell(2, 24)
        surface.boundary_cells = cells[[8, 9, 10, 11]]
        cells = np.append(cells, surface)

        surface = cell(2, 25)
        surface.boundary_cells = cells[[12, 13, 14, 15]]
        cells = np.append(cells, surface)

        volume = cell(3, 26)
        volume.boundary_cells = cells[[20, 21, 22, 23, 24, 25]]
        cells = np.append(cells, volume)
        return cells
