import meshio
import numpy as np
import networkx as nx
from mesh.mesh_cell import MeshCell
from mesh.mesh_cell import barycenter
from mesh.mesher import Mesher
import copy

class Mesh:
    def __init__(self, dimension, file_name):
        self.dimension = dimension
        self.graph = None
        self.cells = np.array([], dtype=MeshCell)
        self.duplicated_ids = {}
        self.conformal_mesh = meshio.read(file_name)
        self.points = self.conformal_mesh.points
        self.cell_data = {} #np.array([], dtype=int)
        self.fracture_normals = {}
        self.Mesher = None

    def set_Mesher(self, Mesher):
        self.Mesher = Mesher

    def mesh_cell_type_index(self, name):
        types = {"vertex": 1, "line": 2, "triangle": 3, "tetra": 4}
        return types[name]

    def validate_entity(self, node_tags):
        perm = np.argsort(node_tags)
        return node_tags[perm]

    def insert_cell_data(self, mesh_cell, type_index, tags, sign=0):

        # composing key
        # The key can be composed in many ways
        chunk = [0 for i in range(6)]
        chunk[0] = sign
        chunk[1] = type_index
        for i, tag in enumerate(tags):
            chunk[i + 2] = tag
        key = ''
        for integer in chunk:
            key = key + str(integer)

        position = self.cell_data.get(key,None) # np.where((self.cell_data == tuple(chunk)).all(axis=1))
        cell_id = None
        if position is None:
            cell_id = len(self.cell_data)
            self.cell_data.__setitem__(key, cell_id)
            self.cells = np.append(self.cells, mesh_cell)
            mesh_cell.set_id(cell_id)
        else:
            cell_id = position
            mesh_cell = self.cells[cell_id]

        return mesh_cell

    def transfer_conformal_mesh(self):

        # preallocate cells objects

        cells = self.conformal_mesh.cells
        physical_tag = self.conformal_mesh.cell_data["gmsh:physical"]
        for cell_block, physical in zip(cells, physical_tag):
            self.insert_simplex_cell(cell_block, physical)

    def insert_simplex_cell(self, cell_block, physical):

        assert self.dimension == 2

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
                    cells_0d.append(mesh_cell.id)

                # 1d cells
                type_index = self.mesh_cell_type_index(cell_block.type)
                mesh_cell = MeshCell(1)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                mesh_cell.set_cells_ids(0,np.array(cells_0d))
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
                    cells_0d.append(mesh_cell.id)

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
                    mesh_cell.set_cells_ids(0,np.array(cells_0d)[con])
                    tags_v = self.validate_entity(node_tags[con])
                    mesh_cell = self.insert_cell_data(mesh_cell,type_index, tags_v)
                    cells_1d.append(mesh_cell.id)

                # 2d cells
                type_index = self.mesh_cell_type_index(cell_block.type)
                mesh_cell = MeshCell(2)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                mesh_cell.set_cells_ids(0,np.array(cells_0d))
                mesh_cell.set_cells_ids(1,np.array(cells_1d))
                tags_v = self.validate_entity(node_tags)
                mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)

    def conformal_mesh_write_vtk(self):

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
        meshio.write("conf_geometric_mesh_2d.vtk", mesh_2d)

        physical_tags_1d = self.conformal_mesh.get_cell_data("gmsh:physical", "line")
        cells_dict = {"line": self.conformal_mesh.get_cells_type("line")}
        cell_data = {"physical_tag": [physical_tags_1d]}
        mesh_1d = meshio.Mesh(
            points=self.conformal_mesh.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("conf_geometric_mesh_1d.vtk", mesh_1d)

        physical_tags_0d = self.conformal_mesh.get_cell_data("gmsh:physical", "vertex")
        cells_dict = {"vertex": self.conformal_mesh.get_cells_type("vertex")}
        cell_data = {"physical_tag": [physical_tags_0d]}
        mesh_0d = meshio.Mesh(
            points=self.conformal_mesh.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("conf_geometric_mesh_0d.vtk", mesh_0d)

    def write_vtk(self):

        assert self.dimension == 2

        # write vtk files
        physical_tags_2d = np.array([cell.material_id for cell in self.cells if
                            cell.dimension == 2])
        entity_tags_2d = np.array([cell.id for cell in self.cells if
                                     cell.dimension == 2])

        con_2d = np.array([cell.node_tags for cell in self.cells if
         cell.dimension == 2])
        cells_dict = {"triangle": con_2d}
        cell_data = {"physical_tag": [physical_tags_2d],
                     "entity_tag": [entity_tags_2d]}
        mesh_2d = meshio.Mesh(
            self.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("geometric_mesh_2d.vtk", mesh_2d)

        physical_tags_1d = np.array([cell.material_id for cell in self.cells if
                            cell.dimension == 1 and cell.material_id != None])
        entity_tags_1d = np.array([cell.id for cell in self.cells if
                                     cell.dimension == 1 and cell.material_id != None])

        con_1d = np.array([cell.node_tags for cell in self.cells if
         cell.dimension == 1 and cell.material_id != None])
        cells_dict = {"line": con_1d}
        cell_data = {"physical_tag": [physical_tags_1d],
                     "entity_tag": [entity_tags_1d]}
        mesh_1d = meshio.Mesh(
            self.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("geometric_mesh_1d.vtk", mesh_1d)


        physical_tags_0d = np.array([cell.material_id for cell in self.cells if
                                     cell.dimension == 0 and cell.material_id != None])
        entity_tags_0d = np.array([cell.id for cell in self.cells if
                                   cell.dimension == 0 and cell.material_id != None])

        con_0d = np.array([cell.node_tags for cell in self.cells if
                           cell.dimension == 0 and cell.material_id != None])
        cells_dict = {"vertex": con_0d}
        cell_data = {"physical_tag": [physical_tags_0d],
                     "entity_tag": [entity_tags_0d]}
        mesh_0d = meshio.Mesh(
            self.points, cells=cells_dict, cell_data=cell_data
        )
        meshio.write("geometric_mesh_0d.vtk", mesh_0d)

    def gather_graph_edges(self, dimension, mesh_cell, tuple_id_list):

        mesh_cell_list = None
        if dimension == 0:
            mesh_cell_list = mesh_cell.cells_ids[0]
        elif dimension == 1:
            mesh_cell_list = mesh_cell.cells_ids[1]
        elif dimension == 2:
            mesh_cell_list = mesh_cell.cells_ids[2]
        else:
            raise ValueError("Dimension not available: ", dimension)

        for id in mesh_cell_list:
            tuple_id_list.append((mesh_cell.id, id))
            if self.cells[id].dimension != dimension:
                self.gather_graph_edges(dimension, self.cells[id], tuple_id_list)


    def build_graph(self, dimension, co_dimension):

        disjoint_cells = [
            cell_i for cell_i in self.cells if cell_i.dimension == dimension
        ]

        tuple_id_list = []
        for cell_i in disjoint_cells:
            self.gather_graph_edges(dimension - co_dimension, cell_i, tuple_id_list)

        graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
        return graph

    def build_graph_on_materials(self, dimension, co_dimension):

        disjoint_cells = [
            cell_i for cell_i in self.cells if cell_i.dimension == dimension and cell_i.material_id is not None
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

        assert self.dimension == 2

        fracture_network = self.Mesher.fracture_network
        graph_nodes = list(fracture_network.graph.nodes())
        geo_cells = fracture_network.cells[graph_nodes]
        i_cells = [cell for cell in geo_cells if
                   cell.dimension == 1 and len(cell.immersed_cells) > 0]
        ni_cells = [cell.id for m_cell in i_cells for cell in m_cell.immersed_cells]
        geo_1_cells = [cell for cell in geo_cells if
                       cell.dimension == 1 and cell.id not in ni_cells]

        for geo_1_cell in geo_1_cells:
            b, e = (geo_1_cell.boundary_cells[0].point_id, geo_1_cell.boundary_cells[1].point_id)
            cell_points = fracture_network.points[[b, e]]
            R = cell_points[1] - cell_points[0]
            tau = (R)/np.linalg.norm(R)
            n = np.array([tau[1], -tau[0]])
            xc = barycenter(cell_points)
            self.fracture_normals[geo_1_cell.id] = (n,xc)



    def cut_conformity_on_fractures(self):

        # this method requires
        # dictionary of fracture id's to (normal,fracture barycenter)
        self.compute_fracture_normals()

        assert self.dimension == 2

        for data in self.fracture_normals.items():

            gd2c1 = self.build_graph(2, 1)
            gd1c1 = self.build_graph(1, 1)

            mat_id, (n, f_xc) = data
            f_cells = [cell for cell in self.cells if cell.material_id == mat_id]
            for mesh_cell in f_cells:
                cell_id = mesh_cell.id
                tag_v = self.validate_entity(mesh_cell.node_tags)

                # cutting edge support
                type_index = self.mesh_cell_type_index('line')
                cells_2d_ids = list(gd2c1.predecessors(cell_id))
                assert len(cells_2d_ids) == 2

                # classify cells
                cell_2d_p = self.cells[cells_2d_ids[0]]
                cell_2d_n = self.cells[cells_2d_ids[1]]
                cell_xc = barycenter(self.points[cell_2d_p.node_tags])[[0,1]]
                negative_q = np.dot(cell_xc - f_xc,n) < 0.0
                if negative_q:
                    cell_2d_p = self.cells[cells_2d_ids[1]]
                    cell_2d_n = self.cells[cells_2d_ids[0]]

                self.create_new_cells(gd1c1, mesh_cell, cell_2d_p, cell_2d_n)


    def create_new_cells(self, frac_graph, d_m_1_frac_cell, cell_p, cell_n):

        # Detecting fracture boundaries
        d_m_1_cells = []
        if self.dimension == 3:
            d_m_1_cells = d_m_1_frac_cell.cells_ids[1]
        else:
            d_m_1_cells = d_m_1_frac_cell.cells_ids[0]

        are_there_boundaries_q = []
        for id in d_m_1_cells:
            pre_ids = list(frac_graph.predecessors(id))
            pre_cells = [id for id in pre_ids if
                              self.cells[id].material_id is not None]
            is_d_m_2_cell_bc_q = len(pre_cells) <= 1
            are_there_boundaries_q.append(is_d_m_2_cell_bc_q)

        if any(are_there_boundaries_q):
            # partial conformity cut
            duplicates_q = [not boundary_q for boundary_q in are_there_boundaries_q]
            d_m_1_cell_p = self.partial_duplicate_cell(d_m_1_frac_cell, +1, duplicates_q)
            d_m_1_cell_n = self.partial_duplicate_cell(d_m_1_frac_cell, -1, duplicates_q)

            d_m_1_cell_id_p = [i for i, id in enumerate(cell_p.cells_ids[1]) if
                         id == d_m_1_frac_cell.id]
            d_m_1_cell_id_n = [i for i, id in enumerate(cell_n.cells_ids[1]) if
                               id == d_m_1_frac_cell.id]

            self.update_codimension_1_cell(cell_p, d_m_1_cell_id_p[0], d_m_1_cell_p)
            self.update_codimension_1_cell(cell_n, d_m_1_cell_id_n[0], d_m_1_cell_n)



            print("Partial duplicated d-1-cells with ids: ", [cell_p.id, cell_n.id])
        else:
            # full conformity cut
            d_m_1_cell_p = self.duplicate_cell(d_m_1_frac_cell, +1)
            d_m_1_cell_n = self.duplicate_cell(d_m_1_frac_cell, -1)

            d_m_1_cell_id_p = [i for i, cell in enumerate(cell_p.cells_1d) if
                         cell.id == d_m_1_frac_cell.id]
            d_m_1_cell_id_n = [i for i, cell in enumerate(cell_n.cells_1d) if
                               cell.id == d_m_1_frac_cell.id]

            self.update_codimension_1_cell(cell_p, d_m_1_cell_id_p[0], d_m_1_cell_p)
            self.update_codimension_1_cell(cell_n, d_m_1_cell_id_n[0], d_m_1_cell_n)

            print("Full duplicated d-1-cells with ids: ", [cell_p.id, cell_n.id])

    def update_codimension_1_cell(self, cell, index, d_m_1_cell):

        if cell.dimension == 3:
            # 3-d case
            assert self.dimension == 2
            # self.update_cells_1d_from_cells_0d(cell, index, d_m_1_cell)
            # current_cell = self.cells_2d[index]
            # for i, cell_0d in enumerate(current_cell.cells_0d):
            #     cell_0d = d_m_1_cell.cells_0d[i]
            #
            # for i, cell_1d in enumerate(current_cell.cells_1d):
            #     cell_1d = d_m_1_cell.cells_1d[i]
            #
            # cellcells_2d[index] = d_m_1_cell
            # self.update_cells_1d_from_cells_0d()
        elif cell.dimension == 2:
            # 2-d case
            self.update_cells_1d_from_cells_0d(cell, index, d_m_1_cell)
            cell.cells_ids[1][index] = d_m_1_cell.id
        elif cell.dimension == 1:
            # 1-d case
            cell.cells_ids[0][index] = d_m_1_cell.id

    def update_cells_1d_from_cells_0d(self, cell, index, d_m_1_cell):

        n_cells_0d = len(cell.cells_ids[0])
        loop = [i for i in range(n_cells_0d)]
        loop.append(loop[0])
        connectivities = np.array(
            [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
        )

        con = connectivities[index]
        for new_id in d_m_1_cell.cells_ids[0]:
            old_id = self.duplicated_ids.get(new_id, None)
            if old_id is None:
                continue
            for c in con:
                current_id = cell.cells_ids[0][c]
                if current_id == old_id:
                    cell.cells_ids[0][c] = new_id

    def duplicate_cell(self, cell, sign):

        mesh_cell = None
        if cell.dimension == 1:
            mesh_cell = self.duplicate_cells_1d(cell, sign)
        elif cell.dimesion == 2:
            mesh_cell = self.duplicate_cells_2d(cell, sign)

        return mesh_cell

    def duplicate_cells_0d(self, cell, sign):

        cells_0d = []
        for d_m_1_cell in cell.cells_ids[0]:
            type_index = self.mesh_cell_type_index('vertex')
            cell_0d = copy.deepcopy(d_m_1_cell)
            node_tags = cell_0d.node_tags
            tags_v = self.validate_entity(node_tags)

            if d_m_1_cell.material_id is not None:
                material_id = - (10*d_m_1_cell.get_material_id() + sign)
                sign = sign * cell.material_id
                cell_0d.set_material_id(material_id)

            cell_0d = self.insert_cell_data(cell_0d, type_index, tags_v, sign)
            cells_0d.append(cell_0d)
            self.duplicated_ids[cell_0d.id] = id

        return cells_0d

    def duplicate_cells_1d(self, cell, sign):

        cells_0d = self.duplicate_cells_0d(cell, sign)
        type_index = self.mesh_cell_type_index('line')
        mesh_cell = MeshCell(1)
        if sign != 0:
            material_id = - (10*cell.get_material_id() + sign)
            mesh_cell.set_material_id(material_id)
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_0d(np.array(cells_0d))
        tags_v = self.validate_entity(cell.node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
        return mesh_cell

    def duplicate_cells_2d(self, cell, sign):

        cells_0d = self.duplicate_cells_0d(cell, sign)

        cells_1d = []
        for d_m_1_cell in cell.cells_1d:
            cell_1d = self.duplicate_cells_1d(d_m_1_cell, sign)
            cells_1d.append(cell_1d)

        type_index = self.mesh_cell_type_index(cell_block.type)
        mesh_cell = MeshCell(2)
        if sign != 0:
            material_id = - (10*cell.get_material_id() + sign)
            mesh_cell.set_material_id(material_id)
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_0d(np.array(cells_0d))
        mesh_cell.set_cells_1d(np.array(cells_1d))
        tags_v = self.validate_entity(node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)
        return mesh_cell

    def partial_duplicate_cell(self, cell, sign, duplicates_q):

        mesh_cell = None
        if cell.dimension == 1:
            mesh_cell = self.partial_duplicate_cells_1d(cell, sign, duplicates_q)
        elif cell.dimesion == 2:
            assert cell.dimesion != 2
            mesh_cell = self.partial_duplicate_cells_2d(cell, sign, duplicates_q, duplicates_q)

        return mesh_cell

    def partial_duplicate_cells_0d(self, cell, sign, duplicates_q):

        cells_0d = []
        for id, duplicate_q in zip(cell.cells_ids[0], duplicates_q):
            d_m_1_cell = self.cells[id]
            type_index = self.mesh_cell_type_index('vertex')
            if duplicate_q:
                cell_0d = copy.deepcopy(d_m_1_cell)
                node_tags = cell_0d.node_tags
                tags_v = self.validate_entity(node_tags)

                if d_m_1_cell.material_id is not None:
                    material_id = - (10*d_m_1_cell.get_material_id() + sign)
                    sign = sign * cell.material_id
                    cell_0d.set_material_id(material_id)

                cell_0d = self.insert_cell_data(cell_0d, type_index, tags_v, sign)
                cells_0d.append(cell_0d.id)
                self.duplicated_ids[cell_0d.id] = id
            else:
                cells_0d.append(d_m_1_cell.id)

        return cells_0d

    def partial_duplicate_cells_1d(self, cell, sign, duplicates_q):

        cells_0d = self.partial_duplicate_cells_0d(cell, sign, duplicates_q)
        type_index = self.mesh_cell_type_index('line')
        mesh_cell = MeshCell(1)
        if sign != 0:
            material_id = - (10*cell.get_material_id() + sign)
            mesh_cell.set_material_id(material_id)
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_ids(0,np.array(cells_0d))
        tags_v = self.validate_entity(cell.node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
        return mesh_cell

    def partial_duplicate_cells_2d(self, cell, sign, duplicates_1d_q, duplicates_0d_q):

        cells_0d = self.partial_duplicate_cells_0d(cell, sign, duplicates_0d_q)
        cells_1d = []
        for d_m_1_cell, duplicate_q in zip(cell.cells_1d,duplicates_1d_q):
            if duplicate_q:
                cell_1d = self.partial_duplicate_cells_1d(d_m_1_cell)
                cells_1d.append(cell_1d)
            else:
                cells_1d.append(d_m_1_cell)

        type_index = self.mesh_cell_type_index(cell_block.type)
        mesh_cell = MeshCell(2)
        if sign != 0:
            material_id = - (10*cell.get_material_id() + sign)
            mesh_cell.set_material_id(material_id)
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_0d(np.array(cells_0d))
        mesh_cell.set_cells_1d(np.array(cells_1d))
        tags_v = self.validate_entity(node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)
        return mesh_cell