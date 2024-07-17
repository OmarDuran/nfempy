import copy
import itertools

import meshio
import networkx as nx
import numpy as np

from topology.domain import Domain
from mesh.mesh_cell import MeshCell, barycenter, rotate_vector
from mesh.mesh_coloring import coloring_mesh_by_co_dimension


class Mesh:
    def __init__(self, dimension, file_name):
        self.dimension = dimension
        self.graph = None
        self.cells = np.array([], dtype=MeshCell)
        self.duplicated_ids = {}
        self.conformal_mesh = meshio.read(file_name)
        self.points = self.conformal_mesh.points
        self.cell_data = {}
        self.fracture_normals = {}
        self.embed_shape_normals = {}
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

    # def replace_cell_data(self, mesh_cell, type_index, tags, sign=0):
    #     # The key can be composed in many ways
    #     chunk = [0 for i in range(6)]
    #     chunk[0] = sign
    #     chunk[1] = type_index
    #     for i, tag in enumerate(tags):
    #         chunk[i + 2] = tag
    #     key = ""
    #     for integer in chunk:
    #         key = key + str(integer)
    #
    #     position = self.cell_data.get(
    #         key, None
    #     )  # np.where((self.cell_data == tuple(chunk)).all(axis=1))
    #     cell_id = None
    #     if position is None:
    #         cell_id = len(self.cell_data)
    #         self.cell_data.__setitem__(key, cell_id)
    #         self.cells = np.append(self.cells, mesh_cell)
    #         mesh_cell.set_id(cell_id)
    #     else:
    #         cell_id = position
    #         mesh_cell = self.cells[cell_id]
    #
    #     return mesh_cell

    # def build_conformal_mesh(self):
    #     # preallocate cells objects
    #
    #     cells = self.conformal_mesh.cells
    #     physical_tag = self.conformal_mesh.cell_data["gmsh:physical"]
    #     for cell_block, physical in zip(cells, physical_tag):
    #         self.insert_simplex_cell_from_block(cell_block, physical)

    def create_cell(self, dimension, node_tags, tag, physical_tag=None):
        mesh_cell = MeshCell(dimension)
        mesh_cell.set_material_id(physical_tag)
        mesh_cell.set_node_tags(node_tags)
        mesh_cell.id = tag
        return mesh_cell

    def max_node_tag(self):
        return self.points.shape[0]

    def append_points(self, points: np.array):
        self.points = np.append(self.points, points, axis=0)

    def max_cell_id(self):
        return self.cells.shape[0]

    def append_cells(self, cells: np.array):
        min_cell_id = np.min(np.array([cell.id for cell in cells]))
        if self.max_cell_id() - 1 >= min_cell_id:
            raise ValueError(
                "cell identifier already used, next cell id available: ",
                self.max_cell_id() - 1,
            )
        self.cells = np.append(self.cells, cells, axis=0)

    def insert_vertex(self, node_tag, physical_tag=None):
        cell_id = self.entities_0d[node_tag]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:
            self.cells[cell_id] = self.create_cell(
                0, np.array([node_tag]), cell_id, physical_tag
            )
            self.cells[cell_id].set_sub_cells_ids(0, np.array([node_tag]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_edge(self, node_tags, physical_tag=None):
        cell_id = self.entities_1d[tuple(np.sort(node_tags))]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:
            vertex_ids = []
            for node_tag in node_tags:
                vertex_id = self.insert_vertex(node_tag)
                vertex_ids.append(vertex_id)

            self.cells[cell_id] = self.create_cell(1, node_tags, cell_id, physical_tag)
            self.cells[cell_id].set_sub_cells_ids(0, np.array(vertex_ids))
            self.cells[cell_id].set_sub_cells_ids(1, np.array([cell_id]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_polygon(self, node_tags, physical_tag=None):
        cell_id = self.entities_2d[tuple(np.sort(node_tags))]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:
            # vertex id
            vertex_ids = [self.entities_0d[node] for node in node_tags]

            edge_0 = np.sort(node_tags[np.array([1, 2])])
            edge_1 = np.sort(node_tags[np.array([0, 2])])
            edge_2 = np.sort(node_tags[np.array([0, 1])])
            edges = [edge_0, edge_1, edge_2]

            edge_ids = []
            for edge in edges:
                edge_id = self.insert_edge(edge)
                edge_ids.append(edge_id)

            # polygonal cells
            self.cells[cell_id] = self.create_cell(2, node_tags, cell_id, physical_tag)

            self.cells[cell_id].set_sub_cells_ids(0, np.array(vertex_ids))
            self.cells[cell_id].set_sub_cells_ids(1, np.array(edge_ids))
            self.cells[cell_id].set_sub_cells_ids(2, np.array([cell_id]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_polyhedron(self, node_tags, physical_tag=None):
        cell_id = self.entities_3d[tuple(np.sort(node_tags))]
        cell_absent_q = self.cells[cell_id] is None
        if cell_absent_q:
            # vertex id
            vertex_ids = [self.entities_0d[node] for node in node_tags]

            edge_0 = [node_tags[2], node_tags[3]]
            edge_1 = [node_tags[1], node_tags[3]]
            edge_2 = [node_tags[1], node_tags[2]]
            edge_3 = [node_tags[0], node_tags[3]]
            edge_4 = [node_tags[0], node_tags[2]]
            edge_5 = [node_tags[0], node_tags[1]]
            edges = [edge_0, edge_1, edge_2, edge_3, edge_4, edge_5]

            edge_ids = []
            for edge in edges:
                edge_id = self.insert_edge(edge)
                edge_ids.append(edge_id)

            face_0 = node_tags[np.array([1, 2, 3])]
            face_1 = node_tags[np.array([0, 2, 3])]
            face_2 = node_tags[np.array([0, 1, 3])]
            face_3 = node_tags[np.array([0, 1, 2])]
            faces = [face_0, face_1, face_2, face_3]
            face_ids = []
            for face in faces:
                face_id = self.insert_polygon(face)
                face_ids.append(face_id)

            # polyhedral cells
            self.cells[cell_id] = self.create_cell(3, node_tags, cell_id, physical_tag)

            self.cells[cell_id].set_sub_cells_ids(0, np.array(vertex_ids))
            self.cells[cell_id].set_sub_cells_ids(1, np.array(edge_ids))
            self.cells[cell_id].set_sub_cells_ids(2, np.array(face_ids))
            self.cells[cell_id].set_sub_cells_ids(3, np.array([cell_id]))
        if physical_tag is not None:
            self.cells[cell_id].set_material_id(physical_tag)
        return cell_id

    def insert_simplex_cell_from_block(self, cell_block, physical):
        if cell_block.dim == 0:
            for node_tags, physical_tag in zip(cell_block.data, physical):
                self.insert_vertex(node_tags[0], physical_tag)

        elif cell_block.dim == 1:
            for node_tags, physical_tag in zip(cell_block.data, physical):
                # node_tags = self.validate_entity(node_tags)
                self.insert_edge(node_tags, physical_tag)

        elif cell_block.dim == 2:
            for node_tags, physical_tag in zip(cell_block.data, physical):
                self.insert_polygon(node_tags, physical_tag)

        elif cell_block.dim == 3:
            for node_tags, physical_tag in zip(cell_block.data, physical):
                self.insert_polyhedron(node_tags, physical_tag)

    def build_conformal_mesh(self):
        # fill node_id to vertices
        vid = 0
        for i, point in enumerate(self.conformal_mesh.points):
            self.entities_0d.__setitem__(i, vid)
            vid += 1

        # at the moment only simplex are supported
        vertex_data = self.conformal_mesh.get_cells_type("vertex")
        line_data = self.conformal_mesh.get_cells_type("line")
        triangle_data = self.conformal_mesh.get_cells_type("triangle")
        tetra_data = self.conformal_mesh.get_cells_type("tetra")

        # fill edge to edge_id from existing tetrahedral
        edge_id = np.max([*self.entities_0d.values()]) + 1
        for nodes in tetra_data:
            # https://defelement.com/elements/lagrange.html
            edge_0 = tuple(np.sort([nodes[2], nodes[3]]))
            edge_1 = tuple(np.sort([nodes[1], nodes[3]]))
            edge_2 = tuple(np.sort([nodes[1], nodes[2]]))
            edge_3 = tuple(np.sort([nodes[0], nodes[3]]))
            edge_4 = tuple(np.sort([nodes[0], nodes[2]]))
            edge_5 = tuple(np.sort([nodes[0], nodes[1]]))
            edges = [edge_0, edge_1, edge_2, edge_3, edge_4, edge_5]

            for edge in edges:
                key_exist_q = self.entities_1d.get(edge, None)
                if key_exist_q is None:
                    self.entities_1d.__setitem__(edge, edge_id)
                    edge_id += 1

        # fill edge to edge_id from existing triangles
        for nodes in triangle_data:
            edge_0 = tuple(np.sort(nodes[np.array([1, 2])]))
            edge_1 = tuple(np.sort(nodes[np.array([0, 2])]))
            edge_2 = tuple(np.sort(nodes[np.array([0, 1])]))
            edges = [edge_0, edge_1, edge_2]
            for edge in edges:
                key_exist_q = self.entities_1d.get(edge, None)
                if key_exist_q is None:
                    self.entities_1d.__setitem__(edge, edge_id)
                    edge_id += 1

        # fill edge to edge_id from existing lines
        for nodes in line_data:
            edge = tuple(np.sort(nodes))
            key_exist_q = self.entities_1d.get(edge, None)
            if key_exist_q is None:
                self.entities_1d.__setitem__(edge, edge_id)
                edge_id += 1

        # fill face to face_id from existing tetrahedral
        face_id = np.max([*self.entities_1d.values()]) + 1
        for nodes in tetra_data:
            face_0 = tuple(np.sort(nodes[np.array([1, 2, 3])]))
            face_1 = tuple(np.sort(nodes[np.array([0, 2, 3])]))
            face_2 = tuple(np.sort(nodes[np.array([0, 1, 3])]))
            face_3 = tuple(np.sort(nodes[np.array([0, 1, 2])]))

            faces = [face_0, face_1, face_2, face_3]
            for face in faces:
                key_exist_q = self.entities_2d.get(face, None)
                if key_exist_q is None:
                    self.entities_2d.__setitem__(face, face_id)
                    face_id += 1

        # fill face to face_id from existing triangles
        for nodes in triangle_data:
            face = tuple(np.sort(nodes))
            key_exist_q = self.entities_2d.get(face, None)
            if key_exist_q is None:
                self.entities_2d.__setitem__(face, face_id)
                face_id += 1

        # fill volume to volume_id from existing tetrahedral
        volume_id = (
            np.max([*self.entities_2d.values()]) + 1
            if len(self.entities_2d) != 0
            else 0
        )
        for nodes in tetra_data:
            volume = tuple(np.sort(nodes))
            key_exist_q = self.entities_3d.get(volume, None)
            if key_exist_q is None:
                self.entities_3d.__setitem__(volume, volume_id)
                volume_id += 1

        n_cells = (
            len(self.entities_0d)
            + len(self.entities_1d)
            + len(self.entities_2d)
            + len(self.entities_3d)
        )
        self.cells = np.empty((n_cells,), dtype=MeshCell)

        cells = self.conformal_mesh.cells
        physical_tag = self.conformal_mesh.cell_data["gmsh:physical"]
        for cell_block, physical in zip(cells, physical_tag):
            self.insert_simplex_cell_from_block(cell_block, physical)

        self.clean_up_entity_maps()

    def clean_up_entity_maps(self):
        self.entities_0d.clear()
        self.entities_1d.clear()
        self.entities_2d.clear()
        self.entities_3d.clear()

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
            print("Number_of_points: ", len(self.points), file=file)
            for i, point in enumerate(self.points):
                print("tag: ", i, " ", *point, sep=" ", file=file)

            print("", file=file)
            print("Cell data", file=file)
            print("Number_of_cells: ", len(self.cells), file=file)
            for cell in self.cells:
                print("Entity_type: ", cell.type, file=file)
                print("Dimension: ", cell.dimension, file=file)
                print("Tag: ", cell.id, file=file)
                print("Physical_tag: ", cell.material_id, file=file)
                print("Physical_name: ", cell.physical_name, file=file)
                print("Node_tags: ", *cell.node_tags, sep=" ", file=file)
                for dim, cells_ids_dim in enumerate(cell.sub_cells_ids):
                    print(
                        "Entities of dimension",
                        dim,
                        ": ",
                        *cells_ids_dim,
                        sep=" ",
                        file=file,
                    )
                print("", file=file)

        file.close()

    def write_vtk(self, coloring_mesh_q=False):
        # write vtk files
        physical_tags_3d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 3 and cell.id is not None
            ]
        )
        entity_tags_3d = np.array(
            [
                cell.id
                for cell in self.cells
                if cell.dimension == 3
                and cell.id is not None
                and cell.material_id is not None
            ]
        )

        con_3d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 3
                and cell.id is not None
                and cell.material_id is not None
            ]
        )

        if len(con_3d) != 0:
            cells_dict = {"tetra": con_3d}
            cell_data = {
                "physical_tag": [physical_tags_3d],
                "entity_tag": [entity_tags_3d],
            }
            if coloring_mesh_q:
                c0_cells = np.array(
                    [
                        cell.id
                        for cell in self.cells
                        if cell.dimension == 3
                        and cell.id is not None
                        and cell.material_id is not None
                    ]
                )

                cells_c1_con_to_color, _ = coloring_mesh_by_co_dimension(8, self, 3, 1)
                cells_c2_con_to_color, _ = coloring_mesh_by_co_dimension(8, self, 3, 2)
                cells_c3_con_to_color, _ = coloring_mesh_by_co_dimension(14, self, 3, 3)
                c1_colors = [cells_c1_con_to_color[id] for id in c0_cells]
                c2_colors = [cells_c2_con_to_color[id] for id in c0_cells]
                c3_colors = [cells_c3_con_to_color[id] for id in c0_cells]
                cell_data.__setitem__("c1_colors", [c1_colors])
                cell_data.__setitem__("c2_colors", [c2_colors])
                cell_data.__setitem__("c3_colors", [c3_colors])

            mesh_3d = meshio.Mesh(self.points, cells=cells_dict, cell_data=cell_data)
            meshio.write("geometric_mesh_3d.vtk", mesh_3d)

        physical_tags_2d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 2
                and cell.id is not None
                and cell.material_id is not None
            ]
        )
        entity_tags_2d = np.array(
            [
                cell.id
                for cell in self.cells
                if cell.dimension == 2
                and cell.id is not None
                and cell.material_id is not None
            ]
        )

        con_2d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 2
                and cell.id is not None
                and cell.material_id is not None
            ]
        )

        if len(con_2d) != 0:
            cells_dict = {"triangle": con_2d}
            cell_data = {
                "physical_tag": [physical_tags_2d],
                "entity_tag": [entity_tags_2d],
            }

            if coloring_mesh_q:
                c0_cells = np.array(
                    [
                        cell.id
                        for cell in self.cells
                        if cell.dimension == 2
                        and cell.id is not None
                        and cell.material_id is not None
                    ]
                )

                cells_c1_con_to_color, _ = coloring_mesh_by_co_dimension(8, self, 2, 1)
                cells_c2_con_to_color, _ = coloring_mesh_by_co_dimension(8, self, 2, 2)
                c1_colors = [cells_c1_con_to_color[id] for id in c0_cells]
                c2_colors = [cells_c2_con_to_color[id] for id in c0_cells]
                cell_data.__setitem__("c1_colors", [c1_colors])
                cell_data.__setitem__("c2_colors", [c2_colors])

            mesh_2d = meshio.Mesh(self.points, cells=cells_dict, cell_data=cell_data)
            meshio.write("geometric_mesh_2d.vtk", mesh_2d)

        physical_tags_1d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 1
                and cell.id is not None
                and cell.material_id is not None
            ]
        )
        entity_tags_1d = np.array(
            [
                cell.id
                for cell in self.cells
                if cell.dimension == 1
                and cell.id is not None
                and cell.material_id is not None
            ]
        )

        con_1d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 1
                and cell.id is not None
                and cell.material_id is not None
            ]
        )
        if len(con_1d) != 0:
            cells_dict = {"line": con_1d}
            cell_data = {
                "physical_tag": [physical_tags_1d],
                "entity_tag": [entity_tags_1d],
            }

            if coloring_mesh_q:
                c0_cells = np.array(
                    [
                        cell.id
                        for cell in self.cells
                        if cell.dimension == 1
                        and cell.id is not None
                        and cell.material_id is not None
                    ]
                )

                cells_c1_con_to_color, _ = coloring_mesh_by_co_dimension(8, self, 1, 1)
                c1_colors = [cells_c1_con_to_color[id] for id in c0_cells]
                cell_data.__setitem__("c1_colors", [c1_colors])

            mesh_1d = meshio.Mesh(self.points, cells=cells_dict, cell_data=cell_data)
            meshio.write("geometric_mesh_1d.vtk", mesh_1d)

        physical_tags_0d = np.array(
            [
                cell.material_id
                for cell in self.cells
                if cell.dimension == 0
                and cell.id is not None
                and cell.material_id is not None
            ]
        )
        entity_tags_0d = np.array(
            [
                cell.id
                for cell in self.cells
                if cell.dimension == 0
                and cell.id is not None
                and cell.material_id is not None
            ]
        )

        con_0d = np.array(
            [
                cell.node_tags
                for cell in self.cells
                if cell.dimension == 0
                and cell.id is not None
                and cell.material_id is not None
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
        if mesh_cell.id is None and mesh_cell.dimension != dimension:
            return

        if dimension in [0, 1, 2, 3]:
            mesh_cell_list = mesh_cell.sub_cells_ids[dimension]
        else:
            raise ValueError("Dimension not available: ", dimension)

        for id in mesh_cell_list:
            b_mesh_cell_index = mesh_cell.index()
            e_mesh_cell_index = self.cells[id].index()
            tuple_id_list.append((b_mesh_cell_index, e_mesh_cell_index))
            if self.cells[id].dimension != dimension:
                self.gather_graph_edges(dimension, self.cells[id], tuple_id_list)

    def build_graph(self, dimension, co_dimension):
        disjoint_cells = [
            cell_i for cell_i in self.cells if cell_i.dimension == dimension
        ]

        tuple_id_list = [[] for i in range(len(disjoint_cells))]
        for i, cell_i in enumerate(disjoint_cells):
            self.gather_graph_edges(dimension - co_dimension, cell_i, tuple_id_list[i])
        tuple_id_list = list(itertools.chain(*tuple_id_list))

        graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
        return graph

    def build_graph_on_physical_tags(self, physical_tags, dimension, co_dimension):
        disjoint_cells = [
            cell_i
            for cell_i in self.cells
            if cell_i.dimension == dimension and cell_i.material_id in physical_tags
        ]

        tuple_id_list = [[] for i in range(len(disjoint_cells))]
        for i, cell_i in enumerate(disjoint_cells):
            self.gather_graph_edges(dimension - co_dimension, cell_i, tuple_id_list[i])
        tuple_id_list = list(itertools.chain(*tuple_id_list))

        graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
        return graph

    def build_graph_from_cell_ids(self, cell_ids, dimension, co_dimension):
        if not isinstance(cell_ids, np.ndarray):
            raise ValueError("Provide cell_ids as a np.ndarray.")
        disjoint_cells = [
            cell_i for cell_i in self.cells[cell_ids] if cell_i.dimension == dimension
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

    def compute_normals_on_embed_shapes(self):
        assert self.dimension == 2
        domain: Domain = self.conformal_mesher.domain
        shapes = domain.shapes[self.dimension]
        assert len(shapes) == 1
        no_immersed_shapes_q = len(shapes[0].immersed_shapes) == 0
        if no_immersed_shapes_q:
            return
        for embed_shape in shapes[0].immersed_shapes:
            b_points = embed_shape.boundary_points()
            R = b_points[1] - b_points[0]
            tau = (R) / np.linalg.norm(R)
            n = rotate_vector(tau, np.pi / 2.0)
            xc = barycenter(b_points)
            self.embed_shape_normals[embed_shape.physical_tag] = (n, xc)

    def cut_conformity_on_embed_shapes(self):
        # As alternative skins can be identified with
        # for the positive side sp_<physical_tag>
        # for the negative side sm_<physical_tag>
        p_tag_scale = 1000

        assert self.dimension == 2

        # dictionary of fracture id's to (normal,fracture barycenter)
        self.compute_normals_on_embed_shapes()

        map_fracs_edge = {}

        for data in self.embed_shape_normals.items():
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
            f_cell_ids = [cell.id for cell in f_cells]
            for i, cell in enumerate(f_cells):
                f_cells[i].normal = n

            # edges duplication
            neighs_by_face = []
            for edge_id in f_cell_ids:
                neighs_by_face = neighs_by_face + list(gd2c1.predecessors(edge_id))
            cells_n = []
            cells_p = []
            for neigh_id in neighs_by_face:
                neigh = self.cells[neigh_id]
                cell_xc = barycenter(self.points[neigh.node_tags])
                positive_q = np.dot(cell_xc - f_xc, n) > 0.0
                if positive_q:
                    cells_p.append(neigh_id)
                else:
                    cells_n.append(neigh_id)

            max_cell_id = len(self.cells)
            new_cell_ids = np.array(list(range(0, 2 * len(f_cells)))) + max_cell_id
            [new_cell_ids_p, new_cell_ids_n] = np.split(new_cell_ids, 2)

            map_edge_p = {}
            for i, cell_id in enumerate(new_cell_ids_p):
                mat_id_p = p_tag_scale * mat_id + 1
                mesh_cell = copy.deepcopy(f_cells[i])
                mesh_cell.set_material_id(mat_id_p)
                mesh_cell.set_physical_name("sp_" + str(mat_id))
                mesh_cell.id = cell_id
                mesh_cell.set_sub_cells_ids(
                    mesh_cell.dimension, np.array([cell_id, f_cells[i].id])
                )
                self.cells = np.append(self.cells, mesh_cell)
                map_edge_p[f_cells[i].id] = cell_id

            map_edge_n = {}
            for i, cell_id in enumerate(new_cell_ids_n):
                mat_id_n = p_tag_scale * mat_id - 1
                mesh_cell = copy.deepcopy(f_cells[i])
                mesh_cell.set_material_id(mat_id_n)
                mesh_cell.set_physical_name("sm_" + str(mat_id))
                mesh_cell.id = cell_id
                mesh_cell.set_sub_cells_ids(
                    mesh_cell.dimension, np.array([cell_id, f_cells[i].id])
                )
                self.cells = np.append(self.cells, mesh_cell)
                map_edge_n[f_cells[i].id] = cell_id

            # collect new edges pair
            for i, cell_id_pair in enumerate(zip(new_cell_ids_p, new_cell_ids_n)):
                map_fracs_edge[f_cells[i].id] = cell_id_pair

            # update edges
            for cell_p_id in cells_p:
                self.update_entity_with_dimension(1, cell_p_id, map_edge_p)

            for cell_n_id in cells_n:
                self.update_entity_with_dimension(1, cell_n_id, map_edge_n)

            # vertices duplication
            vertices = np.unique(
                np.array(
                    [vertex for cell in f_cells for vertex in cell.sub_cells_ids[0]]
                )
            )
            vertices = [
                vertex
                for vertex in vertices
                if self.cells[vertex].material_id != mat_id
            ]
            if len(vertices) == 0:
                continue

            neighs_by_vertex = []
            for vertex in vertices:
                neighs_by_vertex = neighs_by_vertex + list(gd2c2.predecessors(vertex))
                neighs_by_vertex = neighs_by_vertex + list(gd1c1.predecessors(vertex))
            neighs_by_vertex = [
                neigh_id for neigh_id in neighs_by_vertex if neigh_id not in f_cell_ids
            ]
            neighs = np.unique(neighs_by_vertex)
            cells_n = []
            cells_p = []
            for neigh_id in neighs:
                neigh = self.cells[neigh_id]
                cell_xc = barycenter(self.points[neigh.node_tags])
                positive_q = np.dot(cell_xc - f_xc, n) > 0.0
                if positive_q:
                    cells_p.append(neigh_id)
                else:
                    cells_n.append(neigh_id)
            max_node_id = len(self.points)
            new_nodes = np.array(
                [self.points[self.cells[vertex].node_tags[0]] for vertex in vertices]
            )
            self.points = np.vstack((self.points, new_nodes))
            self.points = np.vstack((self.points, new_nodes))
            new_node_tags = np.array(list(range(0, 2 * len(new_nodes)))) + max_node_id

            max_cell_id = len(self.cells)
            new_cell_ids = np.array(list(range(0, 2 * len(new_nodes)))) + max_cell_id
            [new_cell_ids_p, new_cell_ids_n] = np.split(new_cell_ids, 2)
            [new_node_tags_p, new_node_tags_n] = np.split(new_node_tags, 2)

            map_vertex_p = {}
            map_node_p = {}
            for i, cell_id in enumerate(new_cell_ids_p):
                mat_id_p = None
                if self.cells[vertices[i]].material_id is not None:
                    mat_id_p = p_tag_scale * mat_id + 1
                mesh_cell = self.create_cell(
                    0, np.array([new_node_tags_p[i]]), cell_id, mat_id_p
                )
                mesh_cell.set_physical_name("sp_" + str(mat_id))
                mesh_cell.set_sub_cells_ids(
                    mesh_cell.dimension, np.array([new_node_tags_p[i]])
                )
                self.cells = np.append(self.cells, mesh_cell)
                map_vertex_p[vertices[i]] = cell_id
                old_node_tag = self.cells[vertices[i]].node_tags[0]
                map_node_p[old_node_tag] = new_node_tags_p[i]

            map_vertex_n = {}
            map_node_n = {}
            for i, cell_id in enumerate(new_cell_ids_n):
                mat_id_n = None
                if self.cells[vertices[i]].material_id is not None:
                    mat_id_n = p_tag_scale * mat_id - 1

                mesh_cell = self.create_cell(
                    0, np.array([new_node_tags_n[i]]), cell_id, mat_id_n
                )
                mesh_cell.set_physical_name("sm_" + str(mat_id))
                mesh_cell.set_sub_cells_ids(
                    mesh_cell.dimension, np.array([new_node_tags_n[i]])
                )
                self.cells = np.append(self.cells, mesh_cell)
                map_vertex_n[vertices[i]] = cell_id
                old_node_tag = self.cells[vertices[i]].node_tags[0]
                map_node_n[old_node_tag] = new_node_tags_n[i]

            # update nodes and vertex
            for cell_p_id in cells_p:
                self.update_entity_with_dimension(0, cell_p_id, map_vertex_p)
                self.update_nodes_ids(cell_p_id, map_node_p)

            for cell_n_id in cells_n:
                self.update_entity_with_dimension(0, cell_n_id, map_vertex_n)
                self.update_nodes_ids(cell_n_id, map_node_n)

            # update nodes and vertex on duplicated edges
            for cell_p_id in map_edge_p.values():
                self.update_entity_with_dimension(0, cell_p_id, map_vertex_p)
                self.update_nodes_ids(cell_p_id, map_node_p)

            for cell_n_id in map_edge_n.values():
                self.update_entity_with_dimension(0, cell_n_id, map_vertex_n)
                self.update_nodes_ids(cell_n_id, map_node_n)

        return map_fracs_edge

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
            # cell_points = np.hstack((cell_points, np.array([[0.0], [0.0]])))
            R = cell_points[1] - cell_points[0]
            tau = (R) / np.linalg.norm(R)
            n = rotate_vector(tau, np.pi / 2.0)
            xc = barycenter(cell_points)
            self.fracture_normals[geo_1_cell.physical_tag] = (n, xc)

    def apply_visual_opening(self, map_fracs_edge, factor=0.25):
        for data in self.fracture_normals.items():
            mat_id, (n, f_xc) = data
            f_cells = [
                cell
                for cell in self.cells
                if (cell.id is not None)
                and cell.material_id == mat_id
                and cell.dimension == self.dimension - 1
            ]

            # get fracture boundary
            vertices = np.unique(
                np.array(
                    [vertex for cell in f_cells for vertex in cell.sub_cells_ids[0]]
                )
            )
            vertices = [
                vertex
                for vertex in vertices
                if self.cells[vertex].material_id == mat_id
            ]

            # collect node_ids
            node_ids_p = []
            node_ids_n = []
            for cell in f_cells:
                pair = map_fracs_edge[cell.id]
                node_ids_p = node_ids_p + list(self.cells[pair[0]].node_tags)
                node_ids_n = node_ids_n + list(self.cells[pair[1]].node_tags)
            node_ids_p = np.unique(node_ids_p)
            node_ids_n = np.unique(node_ids_n)
            b, e = self.points[vertices]
            lm = np.linalg.norm(e - b)
            for i, pair in enumerate(zip(node_ids_p, node_ids_n)):
                pt_p, pt_n = self.points[pair[0]], self.points[pair[1]]
                s_p = np.linalg.norm(pt_p - b) * np.linalg.norm(pt_p - e) / (lm * lm)
                s_n = np.linalg.norm(pt_n - b) * np.linalg.norm(pt_n - e) / (lm * lm)

                self.points[pair[0]] = self.points[pair[0]] + factor * n * s_p
                self.points[pair[1]] = self.points[pair[1]] - factor * n * s_n

    def cut_conformity_on_fractures_mds_ec(self):
        assert self.dimension == 2
        # dictionary of fracture id's to (normal,fracture barycenter)
        self.compute_fracture_normals()

        map_fracs_edge = {}

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
            f_cell_ids = [cell.id for cell in f_cells]

            neighs_by_face = []
            for edge_id in f_cell_ids:
                neighs_by_face = neighs_by_face + list(gd2c1.predecessors(edge_id))

            # edges duplication
            cells_n = []
            cells_p = []
            for neigh_id in neighs_by_face:
                neigh = self.cells[neigh_id]
                cell_xc = barycenter(self.points[neigh.node_tags])
                positive_q = np.dot(cell_xc - f_xc, n) > 0.0
                if positive_q:
                    cells_p.append(neigh_id)
                else:
                    cells_n.append(neigh_id)

            max_cell_id = len(self.cells)
            new_cell_ids = np.array(list(range(0, 2 * len(f_cells)))) + max_cell_id
            [new_cell_ids_p, new_cell_ids_n] = np.split(new_cell_ids, 2)

            map_edge_p = {}
            for i, cell_id in enumerate(new_cell_ids_p):
                mat_id_p = 10 * mat_id + 1
                mesh_cell = copy.deepcopy(f_cells[i])
                mesh_cell.set_material_id(mat_id_p)
                mesh_cell.id = cell_id
                mesh_cell.set_sub_cells_ids(mesh_cell.dimension, np.array([cell_id]))
                self.cells = np.append(self.cells, mesh_cell)
                map_edge_p[f_cells[i].id] = cell_id

            map_edge_n = {}
            for i, cell_id in enumerate(new_cell_ids_n):
                mat_id_n = 10 * mat_id - 1
                mesh_cell = copy.deepcopy(f_cells[i])
                mesh_cell.set_material_id(mat_id_n)
                mesh_cell.id = cell_id
                mesh_cell.set_sub_cells_ids(mesh_cell.dimension, np.array([cell_id]))
                self.cells = np.append(self.cells, mesh_cell)
                map_edge_n[f_cells[i].id] = cell_id

            # collect new edges pair
            for i, cell_id_pair in enumerate(zip(new_cell_ids_p, new_cell_ids_n)):
                map_fracs_edge[f_cells[i].id] = cell_id_pair

            for cell_p_id in cells_p:
                self.update_entity_with_dimension(1, cell_p_id, map_edge_p)

            for cell_n_id in cells_n:
                self.update_entity_with_dimension(1, cell_n_id, map_edge_n)

            # vertices duplication
            vertices = np.unique(
                np.array(
                    [vertex for cell in f_cells for vertex in cell.sub_cells_ids[0]]
                )
            )
            vertices = [
                vertex
                for vertex in vertices
                if self.cells[vertex].material_id != mat_id
            ]
            if len(vertices) == 0:
                continue

            neighs_by_vertex = []
            for vertex in vertices:
                neighs_by_vertex = neighs_by_vertex + list(gd2c2.predecessors(vertex))
                neighs_by_vertex = neighs_by_vertex + list(gd1c1.predecessors(vertex))
            neighs_by_vertex = [
                neigh_id for neigh_id in neighs_by_vertex if neigh_id not in f_cell_ids
            ]
            neighs = np.unique(neighs_by_vertex)
            cells_n = []
            cells_p = []
            for neigh_id in neighs:
                neigh = self.cells[neigh_id]
                cell_xc = barycenter(self.points[neigh.node_tags])
                positive_q = np.dot(cell_xc - f_xc, n) > 0.0
                if positive_q:
                    cells_p.append(neigh_id)
                else:
                    cells_n.append(neigh_id)
            max_node_id = len(self.points)
            new_nodes = np.array(
                [self.points[self.cells[vertex].node_tags[0]] for vertex in vertices]
            )
            self.points = np.vstack((self.points, new_nodes))
            self.points = np.vstack((self.points, new_nodes))
            new_node_tags = np.array(list(range(0, 2 * len(new_nodes)))) + max_node_id

            max_cell_id = len(self.cells)
            new_cell_ids = np.array(list(range(0, 2 * len(new_nodes)))) + max_cell_id
            [new_cell_ids_p, new_cell_ids_n] = np.split(new_cell_ids, 2)
            [new_node_tags_p, new_node_tags_n] = np.split(new_node_tags, 2)

            map_vertex_p = {}
            map_node_p = {}
            for i, cell_id in enumerate(new_cell_ids_p):
                mat_id_p = None
                if self.cells[vertices[i]].material_id is not None:
                    mat_id_p = 10 * mat_id + 1
                mesh_cell = self.create_cell(
                    0, np.array([new_node_tags_p[i]]), cell_id, mat_id_p
                )
                mesh_cell.set_sub_cells_ids(
                    mesh_cell.dimension, np.array([new_node_tags_p[i]])
                )
                self.cells = np.append(self.cells, mesh_cell)
                map_vertex_p[vertices[i]] = cell_id
                old_node_tag = self.cells[vertices[i]].node_tags[0]
                map_node_p[old_node_tag] = new_node_tags_p[i]

            map_vertex_n = {}
            map_node_n = {}
            for i, cell_id in enumerate(new_cell_ids_n):
                mat_id_n = None
                if self.cells[vertices[i]].material_id is not None:
                    mat_id_n = 10 * mat_id - 1

                mesh_cell = self.create_cell(
                    0, np.array([new_node_tags_n[i]]), cell_id, mat_id_n
                )
                mesh_cell.set_sub_cells_ids(
                    mesh_cell.dimension, np.array([new_node_tags_n[i]])
                )
                self.cells = np.append(self.cells, mesh_cell)
                map_vertex_n[vertices[i]] = cell_id
                old_node_tag = self.cells[vertices[i]].node_tags[0]
                map_node_n[old_node_tag] = new_node_tags_n[i]

            # update nodes and vertex
            for cell_p_id in cells_p:
                self.update_entity_with_dimension(0, cell_p_id, map_vertex_p)
                self.update_nodes_ids(cell_p_id, map_node_p)

            for cell_n_id in cells_n:
                self.update_entity_with_dimension(0, cell_n_id, map_vertex_n)
                self.update_nodes_ids(cell_n_id, map_node_n)

            # update nodes and vertex on duplicated edges
            for cell_p_id in map_edge_p.values():
                self.update_entity_with_dimension(0, cell_p_id, map_vertex_p)
                self.update_nodes_ids(cell_p_id, map_node_p)

            for cell_n_id in map_edge_n.values():
                self.update_entity_with_dimension(0, cell_n_id, map_vertex_n)
                self.update_nodes_ids(cell_n_id, map_node_n)

        return map_fracs_edge

    def cut_conformity_on_fractures(self):
        # this method requires
        # dictionary of fracture id's to (normal,fracture barycenter)
        self.compute_fracture_normals()
        # fracture_tags = self.conformal_mesher.fracture_network.fracture_tags

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

    def update_entity_with_dimension(self, dim, cell_id, entity_map_ids):
        cell = self.cells[cell_id]
        for i, sub_cell_id in enumerate(cell.sub_cells_ids[dim]):
            position = entity_map_ids.get(sub_cell_id, None)
            if position is not None:
                cell.sub_cells_ids[dim][i] = entity_map_ids[sub_cell_id]

    def update_nodes_ids(self, cell_id, node_map_ids):
        cell = self.cells[cell_id]
        for i, node_id in enumerate(cell.node_tags):
            position = node_map_ids.get(node_id, None)
            if position is not None:
                cell.node_tags[i] = node_map_ids[node_id]

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

    def next_d_m_1(self, seed_id, cell_id, cell_m_1_id, graph, closed_q):
        fracture_tags = self.conformal_mesher.fracture_network.fracture_tags
        pc = list(graph.predecessors(cell_m_1_id))
        neighs = [id for id in pc if self.cells[id].material_id not in fracture_tags]
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
        cells_1d = [
            cell.id for cell in self.cells if cell.material_id == fracture_tags[0]
        ]
        f_cells = [self.cells[id] for id in cells_1d if self.cells[id].dimension == 1]

        # get fracture boundary
        f_vertices = np.unique(
            np.array([vertex for cell in f_cells for vertex in cell.sub_cells_ids[0]])
        )
        f_vertices = [
            vertex
            for vertex in f_vertices
            if self.cells[vertex].material_id == fracture_tags[0]
        ]

        graph = self.build_graph_on_materials(1, 1)
        seed_id = f_vertices[0]
        cells_1d = list(graph.predecessors(seed_id))
        skin_cell_ids = [
            id for id in cells_1d if self.cells[id].material_id not in fracture_tags
        ]

        id = seed_id
        skin_cell_id = skin_cell_ids[0]
        self.next_d_m_1(seed_id, skin_cell_id, id, graph, closed_q)
        return closed_q

    def next_d_m_1_cell(
        self, fracture_tags, seed_id, cell_id, cell_m_1_id, graph, closed_q
    ):
        pc = list(graph.predecessors(cell_m_1_id))
        neighs = [id for id in pc if self.cells[id].material_id not in fracture_tags]
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
            # print("cell_dimension  : ", self.cells[fcell_ids[0]].dimension)
            # print("cell_p_name  : ", self.cells[fcell_ids[0]].physical_name)
            # print("cell_xc  : ", barycenter(self.points[self.cells[fcell_ids[0]].node_tags]))
            self.next_d_m_1_cell(
                fracture_tags, seed_id, fcell_ids[0], ids[0], graph, closed_q
            )

    def circulate_internal_bc_from_domain(self):
        assert self.dimension == 2
        domain: Domain = self.conformal_mesher.domain
        shapes = domain.shapes[self.dimension]
        assert len(shapes) == 1
        no_immersed_shapes_q = len(shapes[0].immersed_shapes) == 0
        if no_immersed_shapes_q:
            return True

        closed_q = [False]
        fracture_tags = []
        for embed_shape in shapes[0].immersed_shapes:
            fracture_tags += [embed_shape.physical_tag]

        cells_1d = [
            cell.id for cell in self.cells if cell.material_id == fracture_tags[0]
        ]
        f_cells = [self.cells[id] for id in cells_1d if self.cells[id].dimension == 1]

        # get fracture boundary
        f_vertices = np.unique(
            np.array([vertex for cell in f_cells for vertex in cell.sub_cells_ids[0]])
        )
        f_vertices = [
            vertex
            for vertex in f_vertices
            if self.cells[vertex].material_id == fracture_tags[0]
        ]

        graph = self.build_graph_on_materials(1, 1)
        seed_id = f_vertices[0]
        cells_1d = list(graph.predecessors(seed_id))
        skin_cell_ids = [
            id for id in cells_1d if self.cells[id].material_id not in fracture_tags
        ]

        id = seed_id
        skin_cell_id = skin_cell_ids[0]
        self.next_d_m_1_cell(fracture_tags, seed_id, skin_cell_id, id, graph, closed_q)
        return closed_q
