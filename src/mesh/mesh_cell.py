import abc

import numpy as np


def barycenter(points):
    xc = np.mean(points, axis=0)
    return xc


# class MeshCell(abc.ABC):
class MeshCell:
    def __init__(self, dimension):
        self.dimension = dimension
        self.type = self.mesh_cell_type(dimension)
        self.id = None
        self.material_id = None
        self.node_tags = np.array([], dtype=int)
        self.perm = np.array([], dtype=int)
        data = [np.array([], dtype=int) for i in range(dimension)]
        self.sub_cells_ids = data

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_material_id(self, material_id):
        self.material_id = material_id

    def get_material_id(self):
        return self.material_id

    def set_node_tags(self, node_tags):
        self.node_tags = node_tags
        self.perm = np.argsort(self.node_tags)

    def get_node_tags(self):
        return self.node_tags

    def set_sub_cells_ids(self, dim, cells_ids):
        assert self.dimension > dim
        self.sub_cells_ids[dim] = cells_ids

    # def update_codimension_1_cell(self, index, d_m_1_cell):
    #
    #     # n_cells_0d = len(self.cells_0d)
    #     # loop = [i for i in range(n_cells_0d)]
    #     # loop.append(loop[0])
    #     # connectivities = np.array(
    #     #     [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
    #     # )
    #
    #     if self.dimension == 3:
    #         # 3-d case
    #         assert self.dimension == 2
    #         self.update_cells_1d_from_cells_0d(index, d_m_1_cell)
    #         current_cell = self.cells_2d[index]
    #         for i, cell_0d in enumerate(current_cell.cells_0d):
    #             cell_0d = d_m_1_cell.cells_0d[i]
    #
    #         for i, cell_1d in enumerate(current_cell.cells_1d):
    #             cell_1d = d_m_1_cell.cells_1d[i]
    #
    #         self.cells_2d[index] = d_m_1_cell
    #         self.update_cells_1d_from_cells_0d()
    #     elif self.dimension == 2:
    #         # 2-d case
    #         self.update_cells_1d_from_cells_0d(index, d_m_1_cell)
    #         self.cells_1d[index] = d_m_1_cell
    #     elif self.dimension == 1:
    #         # 1-d case
    #         self.cells_0d[index] = d_m_1_cell

    # def update_cells_1d_from_cells_0d(self, index, d_m_1_cell):
    #
    #     n_cells_0d = len(self.cells_0d)
    #     loop = [i for i in range(n_cells_0d)]
    #     loop.append(loop[0])
    #     connectivities = np.array(
    #         [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
    #     )
    #
    #     con = connectivities[index]
    #     for i, c in enumerate(con):
    #         self.cells_0d[c] = d_m_1_cell.cells_0d[i]
    #
    #     for i, con in enumerate(connectivities):
    #         self.cells_1d[i].cells_0d = self.cells_0d[con]

    @staticmethod
    def mesh_cell_type(dimension):
        types = ("0d-cell", "1d-cell", "2d-cell", "3d-cell")
        return types[dimension]
