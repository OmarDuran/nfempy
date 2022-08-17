import numpy as np
from numpy import linalg as la
import quadpy
import gmsh
import meshio

from shapely.geometry import LineString

import geometry.fracture_network as fn
import networkx as nx

import matplotlib.pyplot as plt

from geometry.cell import Cell
from mesh.mesh_cell import MeshCell


class geometry_builder:

    def __init__(self, dimension):
        self.dimension = dimension
        self.cells = np.array([], dtype=Cell)
        self.points = np.empty((0, dimension), dtype=float)

    def set_boundary(self, vertices, connectivity, material_id):
        self.vertices = vertices
        self.connectivity = connectivity
        self.material_id = material_id

    def build_internal_bc(self, Network, normal_expansion = 1.0e-1):
        assert Network.dimension == self.dimension, f"Geometry and network dimension are not equal {Network.dimension}"

        # classify intersections
        nodes = list(Network.grahp.nodes)
        node_neighs = [[] for _ in nodes]
        for i in range(len(nodes)):
            neighs = list(nx.all_neighbors(Network.grahp, nodes[i]))
            node_neighs[i].append(neighs)

    def gather_graph_edges(self, g_cell: Cell, tuple_id_list):
        for bc_cell in g_cell.boundary_cells:
            tuple_id_list.append((g_cell.id, bc_cell.id))
            if bc_cell.dimension == 0:
                print("BC: Vertex with id: ", bc_cell.id)
            else:
                self.gather_graph_edges(bc_cell, tuple_id_list)
        for immersed_cell in g_cell.immersed_cells:
            tuple_id_list.append((g_cell.id, immersed_cell.id))
            if immersed_cell.dimension == 0:
                print("IM: Vertex with id: ", immersed_cell.id)
            else:
                self.gather_graph_edges(immersed_cell, tuple_id_list)

    def build_grahp(self, all_fixed_d_cells_q=False):

        disjoint_cells = []
        if all_fixed_d_cells_q:
            disjoint_cells = [
                cell_i
                for cell_i in self.cells
            ]
        else:
            disjoint_cells = [
                cell_i
                for cell_i in self.cells if len(cell_i.immersed_cells) == 0
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




def polygon_polygon_intersection():

    fracture_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    fracture_2 = np.array([[0.5, 0., 0.5], [0.5, 0., -0.5], [0.5, 1., -0.5], [0.5, 1., 0.5]])
    fracture_3 = np.array([[0., 0.5, -0.5], [1., 0.5, -0.5], [1., 0.5, 0.5],
     [0., 0.5, 0.5]])

    # fracture_2 = np.array([[0.6, 0., 0.5], [0.6, 0., -0.5], [0.6, 1., -0.5], [0.6, 1., 0.5]])
    # fracture_3 = np.array([[0.25, 0., 0.5], [0.914463, 0.241845, -0.207107], [0.572443, 1.18154, -0.207107],
    #  [-0.0920201, 0.939693, 0.5]])

    fractures = [fracture_1,fracture_2,fracture_3]

    fracture_network = fn.FractureNetwork(dimension=3)
    # fracture_network.render_fractures(fractures)
    fracture_network.intersect_2D_fractures(fractures, True)
    fracture_network.build_grahp()
    fracture_network.draw_grahp()
    ika = 0



# geometry method
def build_box(cells, box_points):

    cells = np.append(cells, np.array([cell(0, i) for i, point in enumerate(box_points)]))

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

def mesh_cell_type_index(name):
    types = {"vertex": 1, "line": 2, "triangle":3, "tetra": 4}
    return types[name]

def main():


    # polygon_polygon_intersection()
    # return 0

    # surface cell
    s = 1.0;
    box_points = s * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    g_builder = geometry_builder(dimension=2)
    g_builder.build_box_2D(box_points)
    g_builder.build_grahp()
    max_point_id = len(g_builder.points)
    max_cell_id = len(g_builder.cells)

    # insert base fractures
    fracture_1 = np.array([[0.25, 0.25], [0.75, 0.75]])
    fracture_2 = np.array([[0.25, 0.75], [0.75, 0.25]])
    fracture_3 = np.array([[0.5, 0.25], [0.5, 0.75]])
    fracture_4 = np.array([[0.65, 0.25], [0.65, 0.75]])
    fracture_5 = np.array([[0.25, 0.5], [0.75, 0.5]])

    fractures = [fracture_1,fracture_2]

    fracture_network = fn.FractureNetwork(dimension=2)
    fracture_network.intersect_1D_fractures(fractures, render_intersection_q = False)
    fracture_network.build_grahp(all_fixed_d_cells_q = True)
    # fracture_network.draw_grahp()
    fracture_network.shift_point_ids(max_point_id)
    fracture_network.shift_cell_ids(max_cell_id)


    import gmsh
    import sys
    gmsh.initialize()
    gmsh.model.add("fn_2d")

    # merged points
    points = g_builder.points
    points = np.append(points,fracture_network.points,axis=0)

    lc = 1.0
    n_points = len(points)
    for tag, point in enumerate(points):
        gmsh.model.geo.addPoint(point[0], point[1], 0, lc, tag + 1)

    # add domain cells
    geo_cells = g_builder.cells[list(g_builder.graph.nodes())]
    geo_1_cells = [cell for cell in geo_cells if cell.dimension == 1]
    geo_2_cells = [cell for cell in geo_cells if cell.dimension == 2]

    tags_2d = []
    for geo_2_cell in geo_2_cells:
        for cell_i in geo_2_cell.boundary_cells:
            b = cell_i.boundary_cells[0].point_id + 1
            e = cell_i.boundary_cells[1].point_id + 1
            gmsh.model.geo.addLine(b, e, cell_i.id)
        tags = [cell.id for cell in geo_2_cell.boundary_cells]
        gmsh.model.geo.addCurveLoop(tags, geo_2_cell.id)
        gmsh.model.geo.addPlaneSurface([geo_2_cell.id],geo_2_cell.id)
        tags_2d.append(geo_2_cell.id)

    gmsh.model.geo.synchronize()

    # add physical tags
    for geo_1_cell in geo_1_cells:
        gmsh.model.addPhysicalGroup(1, [geo_1_cell.id], geo_1_cell.id)

    gmsh.model.addPhysicalGroup(2, tags_2d, tags_2d[0])

    # add fn cells
    geo_cells = fracture_network.cells[list(fracture_network.graph.nodes())]
    geo_0_cells = [cell for cell in geo_cells if cell.dimension == 0]
    geo_1_cells = [cell for cell in geo_cells if cell.dimension == 1 and len(cell.immersed_cells) > 0]

    tags_1d = []
    tags_0d = []
    for geo_1_cell in geo_1_cells:
        n_immersed_cells = len(geo_1_cell.immersed_cells)
        for cell_i in geo_1_cell.immersed_cells:
            b = cell_i.boundary_cells[0].point_id + 1
            e = cell_i.boundary_cells[1].point_id + 1
            gmsh.model.geo.addLine(b, e, cell_i.id)
            tags_1d.append(cell_i.id)
            tags_0d.append(b)
            tags_0d.append(e)

    gmsh.model.geo.synchronize()

    for geo_0_cell in geo_0_cells:
        gmsh.model.addPhysicalGroup(0, [geo_0_cell.point_id + 1], geo_0_cell.id)

    for geo_1_cell in geo_1_cells:
        tags = [cell.id for cell in geo_1_cell.immersed_cells]
        gmsh.model.addPhysicalGroup(1, tags, geo_1_cell.id)

    # embed entities
    gmsh.model.mesh.embed(0, tags_0d, 2, tags_2d[0])
    gmsh.model.mesh.embed(1, tags_1d, 2, tags_2d[0])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("gmesh.msh")

    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    mesh_from_file = meshio.read("gmesh.msh")

    # add skins
    # Clipping out with scissors

    # step 1: create a base geo_mesh
    gm_points = mesh_from_file.points
    gm_cells = np.array([], dtype=MeshCell)
    d_cells = np.empty((0, 5), dtype=int)

    # insert mesh cells
    def validate_entity(node_tags):
        perm = np.argsort(node_tags)
        return node_tags[perm]

    def insert_cell_data(gm_cells, mesh_cell, conn_cells, type_index, tags):
        chunk = [0 for i in range(5)]
        chunk[0] = type_index
        for i, tag in enumerate(tags):
            chunk[i+1] = tag
        position = np.where((conn_cells == tuple(chunk)).all(axis=1))
        cell_id = None
        if position[0].size == 0:
            cell_id = len(conn_cells)
            conn_cells = np.append(conn_cells, [chunk], axis=0)
            gm_cells = np.append(gm_cells,mesh_cell)
        else:
            cell_id = position[0][0]
        mesh_cell.set_id(cell_id)
        return (gm_cells, mesh_cell, conn_cells)

    for cell_block, physical in zip(mesh_from_file.cells,mesh_from_file.cell_data['gmsh:physical']):

        if cell_block.dim == 0: # insert cell
            type_index = mesh_cell_type_index(cell_block.type)
            for node_tags, p_tag in zip(cell_block.data,physical):
                mesh_cell = MeshCell(0)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                tags_v = validate_entity(node_tags)
                gm_cells, mesh_cell, d_cells = insert_cell_data(gm_cells, mesh_cell, d_cells, type_index, tags_v)

        elif cell_block.dim == 1:
            # 0d cells
            for node_tags, p_tag in zip(cell_block.data,physical):
                cells_0d = []
                for node_tag in node_tags:
                    type_index = mesh_cell_type_index('vertex')
                    mesh_cell = MeshCell(0)
                    mesh_cell.set_node_tags(np.array([node_tag]))
                    tag_v = validate_entity(np.array([node_tag]))
                    gm_cells, mesh_cell, d_cells = insert_cell_data(gm_cells, mesh_cell, d_cells, type_index, tag_v)
                    cells_0d.append(mesh_cell)

                # 1d cells
                type_index = mesh_cell_type_index(cell_block.type)
                mesh_cell = MeshCell(1)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                mesh_cell.set_cells_0d(np.array(cells_0d))
                tags_v = validate_entity(node_tags)
                gm_cells, mesh_cell, d_cells = insert_cell_data(gm_cells, mesh_cell, d_cells,
                                                                type_index, tags_v)
        elif cell_block.dim == 2:

            for node_tags in cell_block.data:

                # 0d cells
                cells_0d = []
                for node_tag in node_tags:
                    type_index = mesh_cell_type_index('vertex')
                    mesh_cell = MeshCell(0)
                    mesh_cell.set_node_tags(np.array([node_tag]))
                    tag_v = validate_entity(np.array([node_tag]))
                    gm_cells, mesh_cell, d_cells = insert_cell_data(gm_cells, mesh_cell, d_cells, type_index, tag_v)
                    cells_0d.append(mesh_cell)

                # 1d cells
                loop = [i for i in range(len(node_tags))]
                loop.append(loop[0])
                connectivities = np.array([[loop[index], loop[index + 1]] for index in range(len(loop) - 1)])

                cells_1d = []
                for con in connectivities:
                    type_index = mesh_cell_type_index('line')
                    mesh_cell = MeshCell(1)
                    mesh_cell.set_node_tags(node_tags[con])
                    mesh_cell.set_cells_0d(np.array(cells_0d)[con])
                    tags_v = validate_entity(node_tags[con])
                    gm_cells, mesh_cell, d_cells = insert_cell_data(gm_cells, mesh_cell, d_cells,
                                                                    type_index, tags_v)
                    cells_1d.append(mesh_cell)

                # 2d cells
                type_index = mesh_cell_type_index(cell_block.type)
                mesh_cell = MeshCell(2)
                mesh_cell.set_material_id(p_tag)
                mesh_cell.set_node_tags(node_tags)
                mesh_cell.set_cells_0d(np.array(cells_0d))
                mesh_cell.set_cells_1d(np.array(cells_1d))
                tags_v = validate_entity(node_tags)
                gm_cells, mesh_cell, d_cells = insert_cell_data(gm_cells, mesh_cell, d_cells,
                                                                type_index, tags_v)



    def gather_graph_edges(dimension, mesh_cell: MeshCell, tuple_id_list):
        for cell in mesh_cell.cells_0d:
            tuple_id_list.append((mesh_cell.id, cell.id))
            if cell.dimension == 0:
                print("BC: Vertex with id: ", cell.id)
            else:
                gather_graph_edges(dimension, cell, tuple_id_list)

        for cell in mesh_cell.cells_1d:
            tuple_id_list.append((mesh_cell.id, cell.id))
            if cell.dimension == 0:
                print("BC: Vertex with id: ", cell.id)
            else:
                gather_graph_edges(dimension, cell, tuple_id_list)

        for cell in mesh_cell.cells_2d:
            tuple_id_list.append((mesh_cell.id, cell.id))
            if cell.dimension == 0:
                print("BC: Vertex with id: ", cell.id)
            else:
                gather_graph_edges(dimension, cell, tuple_id_list)

    def build_graph(cells, dimension, co_dimension):

        disjoint_cells = [
            cell_i
            for cell_i in cells
            if cell_i.dimension == dimension
        ]

        tuple_id_list = []
        for cell_i in disjoint_cells:
            gather_graph_edges(co_dimension, cell_i, tuple_id_list)

        graph = nx.from_edgelist(tuple_id_list, create_using=nx.DiGraph)
        return graph

    def draw_graph(graph):
        nx.draw(
            graph,
            pos=nx.circular_layout(graph),
            with_labels=True,
            node_color="skyblue",
        )

    graph = build_graph(gm_cells, 1, 1)
    draw_graph(graph)
    aka = 0

    # def insert_cell(gm_cells,meshio_cell,cell_id):
    #     cell = MeshCell(cell_i.dim, cell_id)
    #     point_ids = cell_i.data
    #     cell.set_nodes(point_ids)
    #     gm_cells = np.append(gm_cells, np.array([cell]))
    #
    # cell_id = 0
    # for cell_i in mesh_from_file.cells:
    #     insert_simplex(gm_cells,cell_i,cell_id)
    #     cell_id = cell_id + 1


    # write vtk files
    physical_tags_2d = mesh_from_file.get_cell_data("gmsh:physical", "triangle")
    cells_dict = {"triangle": mesh_from_file.get_cells_type("triangle")}
    cell_data = {"physical_tag": [physical_tags_2d]}
    mesh_2d = meshio.Mesh(mesh_from_file.points,cells=cells_dict,cell_data=cell_data)
    meshio.write("geometric_mesh_2d.vtk", mesh_2d)

    physical_tags_1d = mesh_from_file.get_cell_data("gmsh:physical", "line")
    cells_dict = {"line": mesh_from_file.get_cells_type("line")}
    cell_data = {"physical_tag": [physical_tags_1d]}
    mesh_1d = meshio.Mesh(mesh_from_file.points, cells=cells_dict, cell_data=cell_data)
    meshio.write("geometric_mesh_1d.vtk", mesh_1d)

    physical_tags_0d = mesh_from_file.get_cell_data("gmsh:physical", "vertex")
    cells_dict = {"vertex": mesh_from_file.get_cells_type("vertex")}
    cell_data = {"physical_tag": [physical_tags_0d]}
    mesh_0d = meshio.Mesh(mesh_from_file.points,cells=cells_dict,cell_data=cell_data)
    meshio.write("geometric_mesh_0d.vtk", mesh_0d)


if __name__ == '__main__':
    main()



