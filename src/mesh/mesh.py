import meshio
import numpy as np
import networkx as nx
from mesh.mesh_cell import MeshCell

from geometry.fracture_network import FractureNetwork

class Mesh:
    def __init__(self, dimension, file_name):
        self.dimension = dimension
        self.graph = None
        self.points = None
        self.cell_data = np.empty((0, 6), dtype=int)
        self.cells = np.array([], dtype=MeshCell)
        self.conformal_mesh = meshio.read(file_name)
        self.fracture_normals = None

    def set_fracture_network(self, fracture_network):
        self.fracture_network = fracture_network

    def mesh_cell_type_index(self, name):
        types = {"vertex": 1, "line": 2, "triangle": 3, "tetra": 4}
        return types[name]

    def validate_entity(self, node_tags):
        perm = np.argsort(node_tags)
        return node_tags[perm]

    def insert_cell_data(self, mesh_cell, type_index, tags, sign=0):
        chunk = [0 for i in range(6)]
        chunk[0] = sign
        chunk[1] = type_index
        for i, tag in enumerate(tags):
            chunk[i + 2] = tag
        position = np.where((self.cell_data == tuple(chunk)).all(axis=1))
        cell_id = None
        if position[0].size == 0:
            cell_id = len(self.cell_data)
            # self.cell_data = np.append(self.cell_data, [chunk], axis=0)
            self.cell_data = np.vstack((self.cell_data,[chunk]))
            self.cells = np.append(self.cells, mesh_cell)
            # self.cells = np.vstack((self.cells,mesh_cell))
            mesh_cell.set_id(cell_id)
        else:
            cell_id = position[0][0]
            mesh_cell = self.cells[cell_id]

        return mesh_cell

    def transfer_conformal_mesh(self):

        cells = self.conformal_mesh.cells
        physical_tag = self.conformal_mesh.cell_data["gmsh:physical"]
        for cell_block, physical in zip(cells, physical_tag):
            self.insert_simplex_cell(cell_block, physical)

    def insert_simplex_cell(self, cell_block, physical):

        if cell_block.dim == 0:
            type_index = self.mesh_cell_type_index(cell_block.type)
            for node_tags, p_tag in zip(cell_block.data, physical):
                mesh_cell = MeshCell(0)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                tags_v = self.validate_entity(node_tags)
                mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)

        elif cell_block.dim == 1:
            # 0d cells
            for node_tags, p_tag in zip(cell_block.data, physical):
                cells_0d = []
                for node_tag in node_tags:
                    type_index = self.mesh_cell_type_index("vertex")
                    mesh_cell = MeshCell(0)
                    mesh_cell.set_node_tags(np.array([node_tag]))
                    tag_v = self.validate_entity(np.array([node_tag]))
                    mesh_cell = self.insert_cell_data(mesh_cell, type_index, tag_v)
                    cells_0d.append(mesh_cell)

                # 1d cells
                type_index = self.mesh_cell_type_index(cell_block.type)
                mesh_cell = MeshCell(1)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                mesh_cell.set_cells_0d(np.array(cells_0d))
                tags_v = self.validate_entity(node_tags)
                mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)
        elif cell_block.dim == 2:

            for node_tags, p_tag in zip(cell_block.data, physical):

                # 0d cells
                cells_0d = []
                for node_tag in node_tags:
                    type_index = self.mesh_cell_type_index("vertex")
                    mesh_cell = MeshCell(0)
                    mesh_cell.set_node_tags(np.array([node_tag]))
                    tag_v = self.validate_entity(np.array([node_tag]))
                    mesh_cell = self.insert_cell_data(mesh_cell, type_index, tag_v)
                    cells_0d.append(mesh_cell)

                # 1d cells
                loop = [i for i in range(len(node_tags))]
                loop.append(loop[0])
                connectivities = np.array(
                    [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
                )

                cells_1d = []
                for con in connectivities:
                    type_index = self.mesh_cell_type_index("line")
                    mesh_cell = MeshCell(1)
                    mesh_cell.set_node_tags(node_tags[con])
                    mesh_cell.set_cells_0d(np.array(cells_0d)[con])
                    tags_v = self.validate_entity(node_tags[con])
                    mesh_cell = self.insert_cell_data(mesh_cell,type_index, tags_v)
                    cells_1d.append(mesh_cell)

                # 2d cells
                type_index = self.mesh_cell_type_index(cell_block.type)
                mesh_cell = MeshCell(2)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                mesh_cell.set_cells_0d(np.array(cells_0d))
                mesh_cell.set_cells_1d(np.array(cells_1d))
                tags_v = self.validate_entity(node_tags)
                mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)

    def write_vtk(self):

        assert self.dimension == 2

        # write vtk files
        physical_tags_2d = self.conformal_mesh.get_cell_data(
            "gmsh:physical", "triangle"
        )
        cells_dict = {"triangle": self.conformal_mesh.get_cells_type("triangle")}
        cell_data = {"physical_tag": [physical_tags_2d]}
        mesh_2d = meshio.Mesh(
            self.conformal_mesh.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("geometric_mesh_2d.vtk", mesh_2d)

        physical_tags_1d = self.conformal_mesh.get_cell_data("gmsh:physical", "line")
        cells_dict = {"line": self.conformal_mesh.get_cells_type("line")}
        cell_data = {"physical_tag": [physical_tags_1d]}
        mesh_1d = meshio.Mesh(
            points=self.conformal_mesh.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("geometric_mesh_1d.vtk", mesh_1d)

        physical_tags_0d = self.conformal_mesh.get_cell_data("gmsh:physical", "vertex")
        cells_dict = {"vertex": self.conformal_mesh.get_cells_type("vertex")}
        cell_data = {"physical_tag": [physical_tags_0d]}
        mesh_0d = meshio.Mesh(
            points=self.conformal_mesh.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("geometric_mesh_0d.vtk", mesh_0d)

    def gather_graph_edges(self, dimension, mesh_cell, tuple_id_list):

        mesh_cell_list = None
        if dimension == 0:
            mesh_cell_list = mesh_cell.cells_0d
        elif dimension == 1:
            mesh_cell_list = mesh_cell.cells_1d
        elif dimension == 2:
            mesh_cell_list = mesh_cell.cells_2d
        else:
            raise ValueError("Dimension not available: ", dimension)

        for cell in mesh_cell_list:
            tuple_id_list.append((mesh_cell.id, cell.id))
            if cell.dimension != 0:
                self.gather_graph_edges(dimension, cell, tuple_id_list)


    def build_graph(self, dimension, co_dimension):

        disjoint_cells = [
            cell_i for cell_i in self.cells if cell_i.dimension == dimension
        ]

        tuple_id_list = []
        for cell_i in disjoint_cells:
            self.gather_graph_edges(dimension - co_dimension, cell_i, tuple_id_list)

        graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
        return graph

    def build_graph_on_index(self, index, dimension, co_dimension):

        disjoint_cells = [
            cell_i
            for cell_i in self.cells
            if cell_i.dimension == dimension and cell_i.id == index
        ]

        tuple_id_list = []
        for cell_i in disjoint_cells:
            gather_graph_edges(dimension - co_dimension, cell_i, tuple_id_list)

        graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
        return graph

    def draw_graph(self, graph):
        nx.draw(
            graph,
            pos=nx.circular_layout(graph),
            with_labels=True,
            node_color="skyblue",
        )

    def compute_fracture_normals(self):
        d = 0

    def cut_conformity_on_fractures(self):

        # this method requires
        # dictionary of fracture id's to normals

        assert self.dimension == 2


        target_mat_id = 13

        f_cells = [cell for cell in gm_cells if cell.material_id == target_mat_id]
        for mesh_cell in f_cells:
            cell_id = mesh_cell.id
            tag_v = validate_entity(mesh_cell.node_tags)

            # cutting edge support
            type_index = mesh_cell_type_index('line')
            cells_2d_ids = list(gd2c1.predecessors(cell_id))
            assert len(cells_2d_ids) == 2
            cell_p = copy.deepcopy(mesh_cell)
            cell_n = copy.deepcopy(mesh_cell)

            # positive side
            gm_cells, cell_p, d_cells = insert_cell_data(gm_cells, cell_p, d_cells,
                                                         type_index, tag_v, +1)
            cell_2d_p = gm_cells[cells_2d_ids[0]]
            edge_id_p = [i for i, cell in enumerate(cell_2d_p.cells_1d) if
                         cell.id == cell_id]
            cell_2d_p.cells_1d[edge_id_p[0]] = cell_p

            # negative side
            gm_cells, cell_n, d_cells = insert_cell_data(gm_cells, cell_n, d_cells,
                                                         type_index, tag_v, -1)

            cell_2d_n = gm_cells[cells_2d_ids[1]]
            edge_id_n = [i for i, cell in enumerate(cell_2d_n.cells_1d) if
                         cell.id == cell_id]
            cell_2d_n.cells_1d[edge_id_n[0]] = cell_n

            print("Edge - New ids: ", [cell_p.id, cell_n.id])

            # cutting node support
            cell_id_0 = mesh_cell.cells_0d[0].id
            cell_id_1 = mesh_cell.cells_0d[1].id
            pre_id_0 = list(gd1c1.predecessors(cell_id_0))
            pre_id_1 = list(gd1c1.predecessors(cell_id_1))

            pre_cells_id_0 = [gm_cells[id] for id in pre_id_0 if
                              gm_cells[id].material_id is not None]
            pre_cells_id_1 = [gm_cells[id] for id in pre_id_1 if
                              gm_cells[id].material_id is not None]
            id_0_bc_q = len(pre_cells_id_0) > 1
            id_1_bc_q = len(pre_cells_id_1) > 1

            if id_0_bc_q:
                mesh_cell_0d = mesh_cell.cells_0d[0]
                cell_id = mesh_cell_0d.id
                cell_0d_p = copy.deepcopy(mesh_cell_0d)
                cell_0d_n = copy.deepcopy(mesh_cell_0d)

                tag_v = validate_entity(mesh_cell_0d.node_tags)
                type_index = mesh_cell_type_index('vertex')

                # positive side
                gm_cells, cell_0d_p, d_cells = insert_cell_data(gm_cells, cell_0d_p,
                                                                d_cells,
                                                                type_index, tag_v, +1)

                vertex_id_p = [i for i, cell in enumerate(cell_p.cells_0d) if
                               cell.id == cell_id]
                cell_p.cells_0d[vertex_id_p[0]] = cell_0d_p

                # negative side
                gm_cells, cell_0d_n, d_cells = insert_cell_data(gm_cells, cell_0d_n,
                                                                d_cells,
                                                                type_index, tag_v, -1)

                vertex_id_n = [i for i, cell in enumerate(cell_n.cells_0d) if
                               cell.id == cell_id]
                cell_n.cells_0d[vertex_id_n[0]] = cell_0d_n
                print("Side 0 - New ids: ", [cell_0d_p.id, cell_0d_n.id])
            else:
                print("Disconnect boundary from skins ")

            if id_1_bc_q:
                mesh_cell_0d = mesh_cell.cells_0d[1]
                cell_id = mesh_cell_0d.id
                cell_0d_p = copy.deepcopy(mesh_cell_0d)
                cell_0d_n = copy.deepcopy(mesh_cell_0d)

                tag_v = validate_entity(mesh_cell_0d.node_tags)
                type_index = mesh_cell_type_index('vertex')

                # positive side
                gm_cells, cell_0d_p, d_cells = insert_cell_data(gm_cells, cell_0d_p,
                                                                d_cells,
                                                                type_index, tag_v, +1)

                vertex_id_p = [i for i, cell in enumerate(cell_p.cells_0d) if
                               cell.id == cell_id]
                cell_p.cells_0d[vertex_id_p[0]] = cell_0d_p

                # negative side
                gm_cells, cell_0d_n, d_cells = insert_cell_data(gm_cells, cell_0d_n,
                                                                d_cells,
                                                                type_index, tag_v, -1)

                vertex_id_n = [i for i, cell in enumerate(cell_n.cells_0d) if
                               cell.id == cell_id]
                cell_n.cells_0d[vertex_id_n[0]] = cell_0d_n
                print("Side 1 - New ids: ", [cell_0d_p.id, cell_0d_n.id])

                cross_fracs = [cell for cell in pre_cells_id_1 if
                               cell.material_id != target_mat_id]
                for cell in cross_fracs:
                    vertex_id_n = [i for i, cell in enumerate(cell.cells_0d) if
                                   cell.id == cell_id]
                    cell.cells_0d[vertex_id_n[0]] = cell_0d_n

            else:
                print("Disconnect boundary from skins ")

            aka = 0
