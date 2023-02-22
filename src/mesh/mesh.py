import meshio
import numpy as np
import networkx as nx
from mesh.mesh_cell import MeshCell
from mesh.mesh_cell import barycenter
from mesh.conformal_mesher import ConformalMesher
import copy


class Mesh:
    def __init__(self, dimension, file_name):
        self.dimension = dimension
        self.graph = None
        self.cells = np.array([], dtype=MeshCell)
        self.duplicated_ids = {}
        self.conformal_mesh = meshio.read(file_name)
        self.points = self.conformal_mesh.points
        self.cell_data = {}  # np.array([], dtype=int)
        self.fracture_normals = {}
        self.conformal_mesher = None

        self.entities_0d = {}
        self.entities_1d = {}
        self.entities_2d = {}
        self.entities_3d = {}


    def set_conformal_mesher(self, conformal_mesher):
        self.conformal_mesher = conformal_mesher

    def mesh_cell_type_index(self, name):
        types = {"vertex": 1, "line": 2, "triangle": 3, "tetra": 4}
        return types[name]

    def validate_entity(self, node_tags):
        perm = np.argsort(node_tags)
        return node_tags[perm]

    def insert_cell_data(self, mesh_cell, type_index, tags, sign=0):

        # The key can be composed in many ways
        chunk = [0 for i in range(6)]
        chunk[0] = sign
        chunk[1] = type_index
        for i, tag in enumerate(tags):
            chunk[i + 2] = tag
        key = ""
        for integer in chunk:
            key = key + str(integer)

        position = self.cell_data.get(key, None)
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

    def replace_cell_data(self, mesh_cell, type_index, tags, sign=0):

        # The key can be composed in many ways
        chunk = [0 for i in range(6)]
        chunk[0] = sign
        chunk[1] = type_index
        for i, tag in enumerate(tags):
            chunk[i + 2] = tag
        key = ""
        for integer in chunk:
            key = key + str(integer)

        position = self.cell_data.get(
            key, None
        )  # np.where((self.cell_data == tuple(chunk)).all(axis=1))
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

    def build_conformal_mesh(self):

        # preallocate cells objects

        cells = self.conformal_mesh.cells
        physical_tag = self.conformal_mesh.cell_data["gmsh:physical"]
        for cell_block, physical in zip(cells, physical_tag):
            self.insert_simplex_cell_from_block(cell_block, physical)

    def create_cell(self, dimension, node_tags, tag, physical_tag):
        mesh_cell = MeshCell(dimension)
        mesh_cell.set_material_id(physical_tag)
        mesh_cell.set_node_tags(node_tags)
        mesh_cell.id = tag
        return mesh_cell

    def insert_vertex(self,node_tag, physical_tag = None):
        cell_id = self.entities_0d[node_tag]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:
            self.cells[cell_id] = self.create_cell(0,np.array([node_tag]), cell_id, physical_tag)
            self.cells[cell_id].set_sub_cells_ids(0, np.array([node_tag]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_edge(self,node_tags,physical_tag = None):
        cell_id = self.entities_1d[tuple(node_tags)]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:
            vertex_ids = []
            for node_tag in node_tags:
                vertex_id = self.insert_vertex(node_tag)
                vertex_ids.append(vertex_id)

            self.cells[cell_id] = self.create_cell(1, node_tags, cell_id,
                                                   physical_tag)
            self.cells[cell_id].set_sub_cells_ids(0, np.array(vertex_ids))
            self.cells[cell_id].set_sub_cells_ids(1, np.array([cell_id]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_polygon(self,node_tags,physical_tag = None):
        cell_id = self.entities_2d[tuple(np.sort(node_tags))]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:

            # vertex id
            vertex_ids = [self.entities_0d[node] for node in node_tags]

            # line loop
            loop = [i for i in range(len(node_tags))]
            loop.append(loop[0])
            connectivity = np.array(
                [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
            )

            edge_ids = []
            for con in connectivity:
                perm = np.argsort(node_tags[con])
                edge_id = self.insert_edge(node_tags[con][perm])
                edge_ids.append(edge_id)

            # polygonal cells
            self.cells[cell_id] = self.create_cell(2, node_tags, cell_id,
                                                   physical_tag)

            self.cells[cell_id].set_sub_cells_ids(0, np.array(vertex_ids))
            self.cells[cell_id].set_sub_cells_ids(1, np.array(edge_ids))
            self.cells[cell_id].set_sub_cells_ids(2, np.array([cell_id]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_simplex_cell_from_block(self, cell_block, physical):

        assert self.dimension == 2

        if cell_block.dim == 0:
            type_index = self.mesh_cell_type_index(cell_block.type)
            for node_tags, physical_tag in zip(cell_block.data, physical):
                self.insert_vertex(node_tags[0],physical_tag)

        elif cell_block.dim == 1:
            for node_tags, physical_tag in zip(cell_block.data, physical):
                # Ensures that all edges are validated
                node_tags = self.validate_entity(node_tags)
                self.insert_edge(node_tags, physical_tag)

        elif cell_block.dim == 2:
            for node_tags, physical_tag in zip(cell_block.data, physical):
                self.insert_polygon(node_tags, physical_tag)

    def build_conformal_mesh_II(self):


        # fill node_id to vertices
        vid = 0
        for i, point in enumerate(self.conformal_mesh.points):
            self.entities_0d.__setitem__(i,vid)
            vid += 1

        # fill node_id to edges
        eid = np.max([*self.entities_0d.values()]) + 1
        for nodes in self.conformal_mesh.get_cells_type("triangle"):
            nodes_ext = np.append(nodes,nodes[0])
            edges = [tuple(np.sort([nodes_ext[i], nodes_ext[i + 1]])) for i, _ in enumerate(nodes, 0)]
            for edge in edges:
                key_exist_q = self.entities_1d.get(edge, None)
                if key_exist_q is None:
                    self.entities_1d.__setitem__(edge, eid)
                    eid += 1

        fid = np.max([*self.entities_1d.values()]) + 1
        for i, nodes in enumerate(self.conformal_mesh.get_cells_type("triangle")):
            face = tuple(np.sort(nodes))
            key_exist_q = self.entities_2d.get(face, None)
            if key_exist_q is None:
                self.entities_2d.__setitem__(face, fid)
                fid += 1

        n_cells = len(self.entities_0d) + len(self.entities_1d) + len(self.entities_2d) + len(self.entities_3d)
        self.cells = np.empty((n_cells,), dtype=MeshCell)

        cells = self.conformal_mesh.cells
        physical_tag = self.conformal_mesh.cell_data["gmsh:physical"]
        for cell_block, physical in zip(cells, physical_tag):
            self.insert_simplex_cell_from_block(cell_block, physical)

        aka = 0

        self.clean_up_entity_maps()

    def clean_up_entity_maps(self):
        self.entities_0d.clear()
        self.entities_1d.clear()
        self.entities_2d.clear()
        self.entities_3d.clear()

    def create_simplex_cell(self, dimension, node_tags, p_tag, sign):

        assert self.dimension == 2

        if dimension == 0:
            type_index = self.mesh_cell_type_index("vertex")
            mesh_cell = MeshCell(0)
            mesh_cell.set_material_id(p_tag)
            mesh_cell.set_node_tags(node_tags)
            tags_v = self.validate_entity(node_tags)
            mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
            return mesh_cell

        elif dimension == 1:
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
            type_index = self.mesh_cell_type_index("line")
            mesh_cell = MeshCell(1)
            mesh_cell.set_material_id(p_tag)
            mesh_cell.set_node_tags(node_tags)
            mesh_cell.set_cells_ids(0, np.array(cells_0d))
            tags_v = self.validate_entity(node_tags)
            mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
            return mesh_cell

        elif dimension == 2:

            # 0d cells
            cells_0d = []
            for node_tag in cell.node_tags:
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
                mesh_cell.set_cells_ids(0, np.array(cells_0d)[con])
                tags_v = self.validate_entity(node_tags[con])
                mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)
                cells_1d.append(mesh_cell.id)

            # 2d cells
            type_index = self.mesh_cell_type_index("triangle")
            mesh_cell = MeshCell(2)
            mesh_cell.set_material_id(p_tag)
            mesh_cell.set_node_tags(node_tags)
            mesh_cell.set_cells_ids(0, np.array(cells_0d))
            mesh_cell.set_cells_ids(1, np.array(cells_1d))
            tags_v = self.validate_entity(node_tags)
            mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
            return mesh_cell

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

    def write_data(self, file_name="gmesh.txt"):
        with open(file_name, "w") as file:

            dimensions = [cell.dimension for cell in self.cells]
            dimensions_with_tag = [
                cell.dimension for cell in self.cells if cell.material_id is not None
            ]

            print("Dimension data", file=file)
            print("Min entity dimension: ", min(dimensions), file=file)
            print("Max entity dimension: ", max(dimensions), file=file)
            print(
                "Min entity dimension with physical tag: ",
                min(dimensions_with_tag),
                file=file,
            )
            print(
                "Max entity dimension with physical tag: ",
                max(dimensions_with_tag),
                file=file,
            )
            print("", file=file)

            print("Point data", file=file)
            print("n_points: ", len(self.points), file=file)
            for i, point in enumerate(self.points):
                print("tag: ", i, " ", *point, sep=" ", file=file)

            print("", file=file)
            print("Cell data", file=file)
            print("n_cells: ", len(self.cells), file=file)
            for cell in self.cells:
                print("Entity_type: ", cell.type, file=file)
                print("Dimension: ", cell.dimension, file=file)
                print("Tag: ", cell.id, file=file)
                print("Physical_tag: ", cell.material_id, file=file)
                print("node_tags: ", *cell.node_tags, sep=" ", file=file)
                for dim, cells_ids_dim in enumerate(cell.sub_cells_ids):
                    print(
                        "Entities of dimension",
                        dim,
                        ": ",
                        *cells_ids_dim,
                        sep=" ",
                        file=file
                    )
                print("", file=file)

        file.close()

    def write_vtk(self):

        assert self.dimension == 2

        # write vtk files
        physical_tags_2d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 2 and cell.id != None
            ]
        )
        entity_tags_2d = np.array(
            [cell.id for cell in self.cells if cell.dimension == 2 and cell.id != None]
        )

        con_2d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 2 and cell.id != None
            ]
        )

        if len(con_2d) != 0:
            cells_dict = {"triangle": con_2d}
            cell_data = {
                "physical_tag": [physical_tags_2d],
                "entity_tag": [entity_tags_2d],
            }
            mesh_2d = meshio.Mesh(self.points, cells=cells_dict, cell_data=cell_data)
            meshio.write("geometric_mesh_2d.vtk", mesh_2d)

        physical_tags_1d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 1 and cell.material_id != None and cell.id != None
            ]
        )
        entity_tags_1d = np.array(
            [
                cell.id
                for cell in self.cells
                if cell.dimension == 1 and cell.material_id != None and cell.id != None
            ]
        )

        con_1d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 1 and cell.material_id != None and cell.id != None
            ]
        )
        if len(con_1d) != 0:
            cells_dict = {"line": con_1d}
            cell_data = {
                "physical_tag": [physical_tags_1d],
                "entity_tag": [entity_tags_1d],
            }
            mesh_1d = meshio.Mesh(self.points, cells=cells_dict, cell_data=cell_data)
            meshio.write("geometric_mesh_1d.vtk", mesh_1d)

        physical_tags_0d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 0 and cell.material_id != None and cell.id != None
            ]
        )
        entity_tags_0d = np.array(
            [
                cell.id
                for cell in self.cells
                if cell.dimension == 0 and cell.material_id != None and cell.id != None
            ]
        )

        con_0d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 0 and cell.material_id != None and cell.id != None
            ]
        )

        if len(con_0d) != 0:
            cells_dict = {"vertex": con_0d}
            cell_data = {
                "physical_tag": [physical_tags_0d],
                "entity_tag": [entity_tags_0d],
            }
            mesh_0d = meshio.Mesh(self.points, cells=cells_dict, cell_data=cell_data)
            meshio.write("geometric_mesh_0d.vtk", mesh_0d)

    def gather_graph_edges(self, dimension, mesh_cell, tuple_id_list):

        if mesh_cell.id is None:
            return

        mesh_cell_list = None
        if dimension == 0:
            mesh_cell_list = mesh_cell.sub_cells_ids[0]
        elif dimension == 1:
            mesh_cell_list = mesh_cell.sub_cells_ids[1]
        elif dimension == 2:
            mesh_cell_list = mesh_cell.sub_cells_ids[2]
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
            cell_i
            for cell_i in self.cells
            if cell_i.dimension == dimension and cell_i.material_id is not None
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
            self.gather_graph_edges(dimension - co_dimension, cell_i, tuple_id_list)

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

        fracture_network = self.conformal_mesher.fracture_network
        graph_nodes = list(fracture_network.graph.nodes())
        geo_cells = fracture_network.cells[graph_nodes]
        i_cells = [
            cell
            for cell in geo_cells
            if cell.dimension == 1 and len(cell.immersed_cells) > 0
        ]
        ni_cells = [cell.id for m_cell in i_cells for cell in m_cell.immersed_cells]
        geo_1_cells = [
            cell
            for cell in geo_cells
            if cell.dimension == 1 and cell.id not in ni_cells
        ]

        for geo_1_cell in geo_1_cells:
            b, e = (
                geo_1_cell.boundary_cells[0].point_id,
                geo_1_cell.boundary_cells[1].point_id,
            )
            cell_points = fracture_network.points[[b, e]]
            R = cell_points[1] - cell_points[0]
            tau = (R) / np.linalg.norm(R)
            n = np.array([tau[1], -tau[0]])
            xc = barycenter(cell_points)
            self.fracture_normals[geo_1_cell.physical_tag] = (n, xc)

    def cut_conformity_on_fractures_mds_ec(self):

        assert self.dimension == 2
        # dictionary of fracture id's to (normal,fracture barycenter)
        self.compute_fracture_normals()

        fracture_tags = self.conformal_mesher.fracture_network.fracture_tags
        for data in self.fracture_normals.items():

            gd2c2 = self.build_graph(2, 2)
            gd2c1 = self.build_graph(2, 1)
            gd1c1 = self.build_graph(1, 1)

            mat_id, (n, f_xc) = data
            f_cells = [
                cell
                for cell in self.cells
                if (cell.id is not None)
                and cell.material_id == mat_id
                and cell.dimension == self.dimension - 1
            ]

            faces = []
            for edge in f_cells:
                faces = faces + list(gd2c1.predecessors(edge.id))

            # edges duplication
            cells_n = []
            cells_p = []
            for face_id in faces:
                face = self.cells[face_id]
                cell_xc = barycenter(self.points[face.node_tags])[[0, 1]]
                positive_q = np.dot(cell_xc - f_xc, n) > 0.0
                if positive_q:
                    cells_p.append(face_id)
                else:
                    cells_n.append(face_id)

            max_cell_id = len(self.cells)
            new_cell_ids = np.array(list(range(0, 2 * len(f_cells)))) + max_cell_id
            [new_cell_ids_p, new_cell_ids_n] = np.split(new_cell_ids, 2)

            map_edge_p = {}
            for i, cell_id in enumerate(new_cell_ids_p):
                mesh_cell = copy.copy(f_cells[i])
                mesh_cell.id = cell_id
                self.cells = np.append(self.cells, mesh_cell)
                map_edge_p[f_cells[i].id] = cell_id

            map_edge_n = {}
            for i, cell_id in enumerate(new_cell_ids_n):
                mesh_cell = copy.copy(f_cells[i])
                mesh_cell.id = cell_id
                self.cells = np.append(self.cells, mesh_cell)
                map_edge_n[f_cells[i].id] = cell_id


            for cell_p_id, cell_n_id  in zip(cells_p,cells_n):
                self.update_entity_with_dimension(1, cell_p_id, map_edge_p)
                self.update_entity_with_dimension(1, cell_n_id, map_edge_n)

            # vertices duplication
            vertices = np.unique(np.array([vertex for cell in f_cells for vertex in cell.sub_cells_ids[0]]))
            vertices = [vertex for vertex in vertices if self.cells[vertex].material_id != mat_id]
            if len(vertices) == 0:
                continue

            faces = []
            for vertex in vertices:
                faces = faces + list(gd2c2.predecessors(vertex))
            faces = np.unique(faces)
            cells_n = []
            cells_p = []
            for face_id in faces:
                face = self.cells[face_id]
                cell_xc = barycenter(self.points[face.node_tags])[[0, 1]]
                positive_q = np.dot(cell_xc - f_xc, n) > 0.0
                if positive_q:
                    cells_p.append(face_id)
                else:
                    cells_n.append(face_id)
            max_node_id = len(self.points)
            new_nodes = np.array([self.points[self.cells[vertex].node_tags[0]] for vertex in vertices])
            self.points = np.vstack((self.points, new_nodes))
            self.points = np.vstack((self.points, new_nodes))
            new_node_tags = np.array(list(range(0, 2 * len(new_nodes)))) + max_node_id

            max_cell_id = len(self.cells)
            new_cell_ids = np.array(list(range(0, 2 * len(new_nodes)))) + max_cell_id
            [new_cell_ids_p, new_cell_ids_n] = np.split(new_cell_ids, 2)

            map_vertex_p = {}
            for i, cell_id in enumerate(new_cell_ids_p):
                mesh_cell = self.create_cell(0, np.array([new_node_tags[i]]), cell_id, mat_id)
                mesh_cell.set_sub_cells_ids(0,np.array([new_node_tags[i]]))
                self.cells = np.append(self.cells,mesh_cell)
                map_vertex_p[vertices[i]] = cell_id

            map_vertex_n = {}
            for i, cell_id in enumerate(new_cell_ids_n):
                mesh_cell = self.create_cell(0, np.array([new_node_tags[i]]), cell_id, mat_id)
                mesh_cell.set_sub_cells_ids(0,np.array([new_node_tags[i]]))
                self.cells = np.append(self.cells,mesh_cell)
                map_vertex_n[vertices[i]] = cell_id

            for cell_p_id, cell_n_id  in zip(cells_p,cells_n):
                self.update_entity_with_dimension(0, cell_p_id, map_vertex_p)
                self.update_entity_with_dimension(0, cell_n_id, map_vertex_n)






    def cut_conformity_on_fractures(self):

        # this method requires
        # dictionary of fracture id's to (normal,fracture barycenter)
        self.compute_fracture_normals()
        fracture_tags = self.conformal_mesher.fracture_network.fracture_tags

        assert self.dimension == 2

        for data in self.fracture_normals.items():

            gd2c2 = self.build_graph(2, 2)
            gd2c1 = self.build_graph(2, 1)
            gd1c1 = self.build_graph(1, 1)

            mat_id, (n, f_xc) = data
            f_cells = [
                cell
                for cell in self.cells
                if (cell.id is not None)
                and cell.material_id == mat_id
                and cell.dimension == 1
            ]
            cells_0d_ids = set([id for cell in f_cells for id in cell.cells_ids[0]])
            cells_0d = [self.cells[id] for id in cells_0d_ids]

            # cutting vertex conformity
            for cell_0d in cells_0d:
                duplicated_ids = {}

                if gd2c2.has_node(cell_0d.id):
                    cells_1d_ids = list(gd1c1.predecessors(cell_0d.id))
                    cells_2d_ids = list(gd2c2.predecessors(cell_0d.id))
                else:
                    continue

                frac_cells_ids = [
                    id
                    for id in cells_1d_ids
                    if self.cells[id].material_id in fracture_tags
                ]

                is_bc_q = cell_0d.material_id == mat_id
                if is_bc_q:
                    # boundary vertex

                    # collect duplicated entities
                    sign = 0
                    physical_tag = mat_id
                    duplicated_ids = self.harvest_duplicates_from_vertex(
                        mat_id,
                        cell_0d,
                        cells_1d_ids,
                        physical_tag,
                        sign,
                        duplicated_ids,
                    )

                    for id in cells_2d_ids:
                        cell_2d = self.cells[id]
                        # update cell_2d
                        self.update_entity_on_dimension(0, cell_2d, duplicated_ids)
                        self.update_entity_on_dimension(
                            1, cell_2d, duplicated_ids, True
                        )
                        for cell_1d_id in cell_2d.cells_ids[1]:
                            cell_1d = self.cells[cell_1d_id]
                            if cell_1d.material_id == mat_id:
                                continue
                            self.update_entity_on_dimension(0, cell_1d, duplicated_ids)

                else:

                    cells_2d_p_ids = []
                    cells_2d_n_ids = []
                    for id in cells_2d_ids:
                        cell_2d = self.cells[id]
                        cell_xc = barycenter(self.points[cell_2d.node_tags])[[0, 1]]
                        negative_q = np.dot(cell_xc - f_xc, n) < 0.0
                        if negative_q:
                            cells_2d_n_ids.append(cell_2d.id)
                        else:
                            cells_2d_p_ids.append(cell_2d.id)

                    cells_p_ids = []
                    cells_n_ids = []
                    for id in cells_1d_ids:
                        cell_1d = self.cells[id]
                        if cell_1d.material_id == mat_id:
                            continue
                        cell_xc = barycenter(self.points[cell_1d.node_tags])[[0, 1]]
                        negative_q = np.dot(cell_xc - f_xc, n) < 0.0
                        if negative_q:
                            cells_n_ids.append(cell_1d.id)
                        else:
                            cells_p_ids.append(cell_1d.id)

                    sign = 0
                    physical_tag = cell_0d.material_id

                    # negative case
                    duplicated_ids = {}
                    duplicated_ids = self.harvest_duplicates_from_vertex(
                        mat_id, cell_0d, cells_n_ids, physical_tag, sign, duplicated_ids
                    )

                    for id in cells_2d_n_ids:
                        cell_2d = self.cells[id]
                        # update cell_2d
                        self.update_entity_on_dimension(0, cell_2d, duplicated_ids)
                        self.update_entity_on_dimension(
                            1, cell_2d, duplicated_ids, True
                        )
                        for cell_1d_id in cell_2d.cells_ids[1]:
                            cell_1d = self.cells[cell_1d_id]
                            if cell_1d.material_id == mat_id:
                                continue
                            self.update_entity_on_dimension(0, cell_1d, duplicated_ids)

                    # positive case
                    duplicated_ids = {}
                    duplicated_ids = self.harvest_duplicates_from_vertex(
                        mat_id, cell_0d, cells_p_ids, physical_tag, sign, duplicated_ids
                    )
                    for id in cells_2d_p_ids:
                        cell_2d = self.cells[id]
                        # update cell_2d
                        self.update_entity_on_dimension(0, cell_2d, duplicated_ids)
                        self.update_entity_on_dimension(
                            1, cell_2d, duplicated_ids, True
                        )
                        for cell_1d_id in cell_2d.cells_ids[1]:
                            cell_1d = self.cells[cell_1d_id]
                            if cell_1d.material_id == mat_id:
                                continue
                            self.update_entity_on_dimension(0, cell_1d, duplicated_ids)

            # cutting edge conformity

            for cell_1d in f_cells:
                cells_2d_ids = list(gd2c1.predecessors(cell_1d.id))

                # create new edge support

                node_tags = self.validate_entity(cell_1d.node_tags)
                for id in cells_2d_ids:
                    cell_2d = self.cells[id]

                    cell_xc = barycenter(self.points[cell_2d.node_tags])[[0, 1]]
                    negative_q = np.dot(cell_xc - f_xc, n) < 0.0
                    sign = 1
                    if negative_q:
                        sign = -1

                    loop = [i for i in range(len(cell_2d.node_tags))]
                    loop.append(loop[0])
                    connectivities = np.array(
                        [
                            [loop[index], loop[index + 1]]
                            for index in range(len(loop) - 1)
                        ]
                    )

                    duplicated_ids = {}
                    for i, cell_id in enumerate(cell_2d.cells_ids[1]):
                        edge_cell = self.cells[cell_id]
                        if (
                            self.validate_entity(edge_cell.node_tags) == node_tags
                        ).all():
                            con = connectivities[i]
                            dim = edge_cell.dimension
                            p_tag = edge_cell.material_id
                            new_cell = self.create_simplex_cell(
                                dim, cell_2d.node_tags[con], p_tag, sign
                            )
                            duplicated_ids[edge_cell.id] = new_cell.id
                    self.update_entity_on_dimension(1, cell_2d, duplicated_ids)

        aka = 0

        # cells_1d = [cell for cell in f_cells if cell.dimension == 1]
        # for mesh_cell in f_cells:
        #     cell_id = mesh_cell.id
        #     tag_v = self.validate_entity(mesh_cell.node_tags)
        #
        #     # cutting edge support
        #     type_index = self.mesh_cell_type_index("line")
        #     cells_2d_ids = list(gd2c1.predecessors(cell_id))
        #     assert len(cells_2d_ids) == 2
        #
        #     # classify cells
        #     cell_2d_p = self.cells[cells_2d_ids[0]]
        #     cell_2d_n = self.cells[cells_2d_ids[1]]
        #     cell_xc = barycenter(self.points[cell_2d_p.node_tags])[[0, 1]]
        #     negative_q = np.dot(cell_xc - f_xc, n) < 0.0
        #     if negative_q:
        #         cell_2d_p = self.cells[cells_2d_ids[1]]
        #         cell_2d_n = self.cells[cells_2d_ids[0]]
        #
        #     self.create_new_cells(gd1c1, mesh_cell, cell_2d_p, cell_2d_n)
        #
        # self.duplicated_ids = {}

    def duplicate_vertice(self, node_tag):
        vertice = self.points[node_tag]
        self.points = np.append(self.points, [vertice], axis=0)
        node_tag = np.array([len(self.points) - 1])
        return node_tag

    def duplicate_entity(
        self, name, cell, node_tags, physical_tag, sign, duplicated_ids
    ):
        type_index = self.mesh_cell_type_index(name)
        new_cell = copy.deepcopy(cell)
        new_cell.node_tags = node_tags
        tags_v = self.validate_entity(node_tags)
        new_cell.set_material_id(physical_tag)
        new_cell = self.insert_cell_data(new_cell, type_index, tags_v, sign)
        duplicated_ids[cell.id] = new_cell.id
        return new_cell, duplicated_ids

    def harvest_duplicates_from_vertex(
        self, frac_p_tag, cell, cells_1d_ids, physical_tag, sign, duplicated_ids
    ):

        vertex_node_tag = self.duplicate_vertice(cell.node_tags[0])
        _, duplicated_ids = self.duplicate_entity(
            "vertex",
            cell,
            vertex_node_tag,
            physical_tag,
            sign,
            duplicated_ids,
        )

        for id in cells_1d_ids:
            if self.cells[id].material_id == frac_p_tag:
                continue

            node_tags = self.cells[id].node_tags
            for i, node_tag in enumerate(node_tags):
                if node_tag == cell.node_tags[0]:
                    node_tags[i] = vertex_node_tag[0]

            physical_tag = self.cells[id].material_id
            _, duplicated_ids = self.duplicate_entity(
                "line",
                self.cells[id],
                node_tags,
                physical_tag,
                sign,
                duplicated_ids,
            )

        return duplicated_ids

    def update_entity_with_dimension(
        self, dim, cell_id, entity_map_ids
    ):
        cell = self.cells[cell_id]
        for i, sub_cell_id in enumerate(cell.sub_cells_ids[dim]):
            position = entity_map_ids.get(sub_cell_id, None)
            if position is not None:
                cell.sub_cells_ids[dim][i] = entity_map_ids[sub_cell_id]
        if cell.dimension > dim+1:
            for sub_cell_id in cell.sub_cells_ids[dim+1]:
                self.update_entity_with_dimension(dim,sub_cell_id,entity_map_ids)




    def update_entity_on_dimension(
        self, dim, cell, duplicated_ids, invalidate_old_q=False
    ):

        for i, cell_id in enumerate(cell.cells_ids[dim]):
            position = duplicated_ids.get(cell_id, None)
            if position is not None:
                cell.cells_ids[dim][i] = duplicated_ids[cell_id]
                # update node tags
                if dim == 0:
                    vertex_id = cell.cells_ids[dim][i]
                    cell.node_tags[i] = self.cells[vertex_id].node_tags[0]
                if invalidate_old_q:
                    self.cells[cell_id].id = None

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
            pre_cells = [id for id in pre_ids if self.cells[id].material_id is not None]
            is_d_m_2_cell_bc_q = len(pre_cells) <= 1
            are_there_boundaries_q.append(is_d_m_2_cell_bc_q)

        if any(are_there_boundaries_q):
            mat_id = d_m_1_frac_cell.material_id
            # partial conformity cut
            d_m_1_cell_id_p = [
                i
                for i, id in enumerate(cell_p.cells_ids[1])
                if id == d_m_1_frac_cell.id
            ]
            d_m_1_cell_id_n = [
                i
                for i, id in enumerate(cell_n.cells_ids[1])
                if id == d_m_1_frac_cell.id
            ]

            duplicates_q = [not boundary_q for boundary_q in are_there_boundaries_q]
            d_m_1_cell_p = self.partial_duplicate_cell(
                mat_id, self.cells[d_m_1_frac_cell.id], +1, duplicates_q
            )
            d_m_1_cell_n = self.partial_duplicate_cell(
                mat_id, self.cells[d_m_1_frac_cell.id], -1, duplicates_q
            )

            self.update_codimension_1_cell(cell_p, d_m_1_cell_id_p[0], d_m_1_cell_p)
            self.update_codimension_1_cell(cell_n, d_m_1_cell_id_n[0], d_m_1_cell_n)

            # print("Partial duplicated d-1-cells with ids: ", [cell_p.id, cell_n.id])
        else:
            # full conformity cut
            mat_id = d_m_1_frac_cell.material_id
            d_m_1_cell_id_p = [
                i
                for i, id in enumerate(cell_p.cells_ids[1])
                if id == d_m_1_frac_cell.id
            ]
            d_m_1_cell_id_n = [
                i
                for i, id in enumerate(cell_n.cells_ids[1])
                if id == d_m_1_frac_cell.id
            ]

            d_m_1_cell_p = self.duplicate_cell(
                mat_id, self.cells[d_m_1_frac_cell.id], +1
            )
            d_m_1_cell_n = self.duplicate_cell(
                mat_id, self.cells[d_m_1_frac_cell.id], -1
            )

            self.update_codimension_1_cell(cell_p, d_m_1_cell_id_p[0], d_m_1_cell_p)
            self.update_codimension_1_cell(cell_n, d_m_1_cell_id_n[0], d_m_1_cell_n)

            # print("Full duplicated d-1-cells with ids: ", [cell_p.id, cell_n.id])

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

        # Update 0d cells
        con = connectivities[index]
        for new_id in d_m_1_cell.cells_ids[0]:
            old_id = self.duplicated_ids.get(new_id, None)
            if old_id is None:
                continue
            for c in con:
                current_id = cell.cells_ids[0][c]
                if current_id == old_id:
                    cell.cells_ids[0][c] = new_id

        # update 1d cells
        cells_1d = [self.cells[i] for i in cell.cells_ids[1]]
        for new_id in d_m_1_cell.cells_ids[0]:
            old_id = self.duplicated_ids.get(new_id, None)
            if old_id is None:
                continue
            for cell_1d in cells_1d:
                for i, id in enumerate(cell_1d.cells_ids[0]):
                    if id == old_id:
                        cell_1d.cells_ids[0][i] = new_id

    def duplicate_cell(self, mat_id, cell, sign):

        mesh_cell = None
        if cell.dimension == 1:
            mesh_cell = self.duplicate_cells_1d(mat_id, cell, sign)
        elif cell.dimesion == 2:
            assert cell.dimesion != 2
            mesh_cell = self.duplicate_cells_2d(cell, sign)

        return mesh_cell

    def duplicate_cells_0d(self, cell, sign):

        cells_0d = []
        for id in cell.cells_ids[0]:
            d_m_1_cell = self.cells[id]
            type_index = self.mesh_cell_type_index("vertex")
            cell_0d = copy.deepcopy(d_m_1_cell)
            node_tags = cell_0d.node_tags
            tags_v = self.validate_entity(node_tags)

            mat_id = self.cells[id].material_id
            if mat_id is not None:
                material_id = sign * (1000 * mat_id + sign)
                cell_0d.set_material_id(material_id)
                sign = material_id

            cell_0d = self.insert_cell_data(cell_0d, type_index, tags_v, sign)
            cells_0d.append(cell_0d.id)
            self.duplicated_ids[cell_0d.id] = id

        return cells_0d

    def duplicate_cells_1d(self, mat_id, cell, sign):

        cells_0d = self.duplicate_cells_0d(cell, sign)
        type_index = self.mesh_cell_type_index("line")
        mesh_cell = MeshCell(1)
        if sign != 0:
            material_id = sign * (1000 * mat_id + sign)
            mesh_cell.set_material_id(material_id)
            sign = material_id
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_ids(0, np.array(cells_0d))
        tags_v = self.validate_entity(cell.node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
        self.duplicated_ids[mesh_cell.id] = cell.id
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
            material_id = -(10 * cell.get_material_id() + sign)
            mesh_cell.set_material_id(material_id)
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_0d(np.array(cells_0d))
        mesh_cell.set_cells_1d(np.array(cells_1d))
        tags_v = self.validate_entity(node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)
        return mesh_cell

    def partial_duplicate_cell(self, mat_id, cell, sign, duplicates_q):

        mesh_cell = None
        if cell.dimension == 1:
            mesh_cell = self.partial_duplicate_cells_1d(
                mat_id, cell, sign, duplicates_q
            )
        elif cell.dimesion == 2:
            assert cell.dimesion != 2
            mesh_cell = self.partial_duplicate_cells_2d(
                mat_id, cell, sign, duplicates_q, duplicates_q
            )

        return mesh_cell

    def partial_duplicate_cells_0d(self, cell, sign, duplicates_q):

        cells_0d = []
        for id, duplicate_q in zip(cell.cells_ids[0], duplicates_q):
            d_m_1_cell = self.cells[id]
            type_index = self.mesh_cell_type_index("vertex")
            if duplicate_q:
                cell_0d = copy.deepcopy(d_m_1_cell)
                node_tags = cell_0d.node_tags
                tags_v = self.validate_entity(node_tags)

                mat_id = self.cells[id].material_id
                if mat_id is not None:
                    material_id = sign * (1000 * mat_id + sign)
                    cell_0d.set_material_id(material_id)
                    sign = material_id

                cell_0d = self.insert_cell_data(cell_0d, type_index, tags_v, sign)
                cells_0d.append(cell_0d.id)
                self.duplicated_ids[cell_0d.id] = id
            else:
                cells_0d.append(d_m_1_cell.id)

        return cells_0d

    def partial_duplicate_cells_1d(self, mat_id, cell, sign, duplicates_q):

        cells_0d = self.partial_duplicate_cells_0d(cell, sign, duplicates_q)
        type_index = self.mesh_cell_type_index("line")
        mesh_cell = MeshCell(1)
        if sign != 0:
            material_id = sign * (1000 * mat_id + sign)
            mesh_cell.set_material_id(material_id)
            sign = material_id
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_ids(0, np.array(cells_0d))
        tags_v = self.validate_entity(cell.node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v, sign)
        self.duplicated_ids[mesh_cell.id] = cell.id
        return mesh_cell

    def partial_duplicate_cells_2d(
        self, mat_id, cell, sign, duplicates_1d_q, duplicates_0d_q
    ):

        cells_0d = self.partial_duplicate_cells_0d(mat_id, cell, sign, duplicates_0d_q)
        cells_1d = []
        for d_m_1_cell, duplicate_q in zip(cell.cells_1d, duplicates_1d_q):
            if duplicate_q:
                cell_1d = self.partial_duplicate_cells_1d(d_m_1_cell)
                cells_1d.append(cell_1d)
            else:
                cells_1d.append(d_m_1_cell)

        type_index = self.mesh_cell_type_index(cell_block.type)
        mesh_cell = MeshCell(2)
        if sign != 0:
            material_id = -(10 * cell.get_material_id() + sign)
            mesh_cell.set_material_id(material_id)
        mesh_cell.set_node_tags(cell.node_tags)
        mesh_cell.set_cells_0d(np.array(cells_0d))
        mesh_cell.set_cells_1d(np.array(cells_1d))
        tags_v = self.validate_entity(node_tags)
        mesh_cell = self.insert_cell_data(mesh_cell, type_index, tags_v)
        return mesh_cell

    def next_d_m_1(self, seed_id, cell_id, cell_m_1_id, graph, closed_q):

        fracture_tags = self.conformal_mesher.fracture_network.fracture_tags
        pc = list(graph.predecessors(cell_m_1_id))
        neighs = [id for id in pc if self.cells[id].material_id in fracture_tags]
        assert len(neighs) == 2

        fcell_ids = [id for id in neighs if id != cell_id]
        assert len(fcell_ids) == 1

        sc = list(graph.successors(fcell_ids[0]))
        ids = [s_id for s_id in sc if s_id != cell_m_1_id]
        assert len(ids) == 1
        if seed_id == ids[0]:
            # print("Seed id was found: ", ids[0])
            # print("Skin boundary is closed.")
            closed_q[0] = True
        else:
            # print("Next pair:")
            # print("cell_id      : ", fcell_ids[0])
            # print("cell_m_1_id  : ", ids[0])
            self.next_d_m_1(seed_id, fcell_ids[0], ids[0], graph, closed_q)

    def circulate_internal_bc(self):

        closed_q = [False]
        fracture_tags = self.conformal_mesher.fracture_network.fracture_tags
        graph_e_to_cell = self.build_graph_on_materials(2, 1)
        cells_1d = [
            cell.id for cell in self.cells if cell.material_id == fracture_tags[0]
        ]
        f_cells = [id for id in cells_1d if graph_e_to_cell.has_node(id)]

        cell_1d = self.cells[f_cells[0]]
        assert cell_1d.dimension == 1
        graph = self.build_graph_on_materials(1, 1)
        seed_id = cell_1d.sub_cells_ids[0][0]

        id = seed_id
        fcell_id = cell_1d.id
        self.next_d_m_1(seed_id, fcell_id, id, graph, closed_q)
        return closed_q
