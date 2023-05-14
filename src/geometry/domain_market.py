import numpy as np

from geometry.vertex import Vertex
from geometry.edge import Edge
from geometry.wire import Wire
from geometry.face import Face
from geometry.shell import Shell
from geometry.solid import Solid
from geometry.solid import Shape
from geometry.domain import Domain
from geometry.shape_manipulation import ShapeManipulation
import csv


def read_fractures_file(n_points, file_name):
    fractures = np.empty((0, n_points, 3), float)
    with open(file_name, "r") as file:
        loaded = csv.reader(file)
        for line in loaded:
            frac = [float(val) for val in line]
            fractures = np.append(
                fractures, np.array([np.split(np.array(frac), n_points)]), axis=0
            )
    return fractures


def build_box_1D(box_points, physical_tags=None):
    if physical_tags is None:
        physical_tags = {"line": 1, "bc_0": 2, "bc_1": 3}

    domain = Domain(dimension=1)
    vertices = np.array([Vertex(tag, point) for tag, point in enumerate(box_points)])
    domain.append_shapes(vertices)

    domain.shapes[0][0].physical_tag = physical_tags.get("bc_0", None)
    domain.shapes[0][1].physical_tag = physical_tags.get("bc_1", None)

    shape_tag = 0
    vertex_indices = [0, 1]
    edge = Edge(shape_tag, domain.shapes[0][vertex_indices])
    edge.physical_tag = physical_tags.get("line", None)
    domain.append_shapes(np.array([edge]))

    return domain


def build_box_2D(box_points, physical_tags=None):
    if physical_tags is None:
        physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}

    domain = Domain(dimension=2)
    vertices = np.array([Vertex(tag, point) for tag, point in enumerate(box_points)])
    domain.append_shapes(vertices)

    loop = [i for i in range(len(box_points))]
    loop.append(loop[0])
    edges_connectivities = np.array(
        [[loop[index], loop[index + 1]] for index in range(len(loop) - 1)]
    )

    edges_list = []
    bc_id = 0
    bc_name = "bc_"
    for tag, edge_con in enumerate(edges_connectivities):
        edge = Edge(tag, domain.shapes[0][edge_con])
        edge.physical_tag = physical_tags.get(bc_name + str(bc_id), None)
        bc_id += 1
        domain.shapes[1] = np.append(domain.shapes[1], edge)
        edges_list.append(edge)

    vertices = np.array([domain.shapes[0][loop[0]], domain.shapes[0][loop[-1]]])
    tag += 1
    wire = Wire(tag, np.array(edges_list), vertices)
    domain.append_shapes(np.array([wire]))

    surface = Face(0, np.array([wire]))
    surface.physical_tag = physical_tags.get("area", None)
    domain.append_shapes(np.array([surface]))

    return domain


def build_box_3D(box_points, physical_tags=None):
    if physical_tags is None:
        physical_tags = {
            "solid": 1,
            "bc_0": 2,
            "bc_1": 3,
            "bc_2": 4,
            "bc_3": 5,
            "bc_4": 6,
            "bc_5": 7,
        }

    domain = Domain(dimension=3)
    vertices = np.array([Vertex(tag, point) for tag, point in enumerate(box_points)])
    domain.append_shapes(vertices)

    edge_connectivities = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    for tag, con in enumerate(edge_connectivities):
        edge = Edge(tag, domain.shapes[0][con])
        domain.shapes[1] = np.append(domain.shapes[1], edge)

    tag += 1
    wire_0 = Wire(tag, domain.shapes[1][[0, 1, 2, 3]], domain.shapes[0][[0]])
    domain.append_shapes(np.array([wire_0]))

    tag += 1
    wire_1 = Wire(tag, domain.shapes[1][[4, 5, 6, 7]], domain.shapes[0][[4]])
    domain.append_shapes(np.array([wire_1]))

    tag += 1
    wire_2 = Wire(tag, domain.shapes[1][[0, 9, 4, 8]], domain.shapes[0][[0]])
    domain.append_shapes(np.array([wire_2]))

    tag += 1
    wire_3 = Wire(tag, domain.shapes[1][[1, 10, 5, 9]], domain.shapes[0][[1]])
    domain.append_shapes(np.array([wire_3]))

    tag += 1
    wire_4 = Wire(tag, domain.shapes[1][[10, 6, 11, 2]], domain.shapes[0][[2]])
    domain.append_shapes(np.array([wire_4]))

    tag += 1
    wire_5 = Wire(tag, domain.shapes[1][[3, 11, 7, 8]], domain.shapes[0][[3]])
    domain.append_shapes(np.array([wire_5]))

    tag += 1
    surface = Face(tag, np.array([wire_0]))
    surface.physical_tag = physical_tags.get("bc_0", None)
    domain.append_shapes(np.array([surface]))

    tag += 1
    surface = Face(tag, np.array([wire_1]))
    surface.physical_tag = physical_tags.get("bc_1", None)
    domain.append_shapes(np.array([surface]))

    tag += 1
    surface = Face(tag, np.array([wire_2]))
    surface.physical_tag = physical_tags.get("bc_2", None)
    domain.append_shapes(np.array([surface]))

    tag += 1
    surface = Face(tag, np.array([wire_3]))
    surface.physical_tag = physical_tags.get("bc_3", None)
    domain.append_shapes(np.array([surface]))

    tag += 1
    surface = Face(tag, np.array([wire_4]))
    surface.physical_tag = physical_tags.get("bc_4", None)
    domain.append_shapes(np.array([surface]))

    tag += 1
    surface = Face(tag, np.array([wire_5]))
    surface.physical_tag = physical_tags.get("bc_5", None)
    domain.append_shapes(np.array([surface]))

    tag += 1
    shell = Shell(
        tag,
        domain.shapes[2],
        np.array([wire_0, wire_1, wire_2, wire_3, wire_4, wire_5]),
    )
    domain.append_shapes(np.array([shell]))

    tag += 1
    solid = Solid(tag, np.array([shell]))
    solid.physical_tag = physical_tags.get("solid", None)
    domain.append_shapes(np.array([solid]))

    return domain


def build_disjoint_lines(file_name, max_e_tag=0, max_v_tag=0, max_p_tag=0):
    domain = Domain(dimension=2)
    lines = read_fractures_file(2, file_name)
    physical_tags = [i + max_p_tag for i in range(len(lines))]
    v_tag = max_v_tag
    e_tag = max_e_tag
    for line, physical_tag in zip(lines, physical_tags):
        v0 = Vertex(v_tag, line[0])
        v_tag += 1
        v1 = Vertex(v_tag, line[1])
        v_tag += 1
        e = Edge(e_tag, np.array([v0, v1]))
        e.physical_tag = physical_tag
        e_tag += 1
        domain.append_shapes(np.array([v0, v1]))
        domain.append_shapes(np.array([e]))

    return domain


def build_box_2D_with_lines(box_points, lines_file, physical_tags=None):
    if physical_tags is None:
        physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}

    domain = build_box_2D(box_points, physical_tags)
    max_v_tag = len(domain.shapes[0])
    max_e_tag = len(domain.shapes[1])
    max_p_tag = domain.max_physical_tag()
    face = domain.shapes[2][0]

    # lines
    domain_lines = build_disjoint_lines(
        lines_file, max_e_tag=max_e_tag, max_v_tag=max_v_tag, max_p_tag=max_p_tag + 1
    )

    ShapeManipulation.embed_edge_in_face(domain_lines.shapes[1], face)
    domain.append_shapes(domain_lines.shapes[0])
    domain.append_shapes(domain_lines.shapes[1])

    max_v_tag = len(domain.shapes[0])
    max_e_tag = len(domain.shapes[1])
    max_p_tag = domain.max_physical_tag()

    # performing multiple intersection of connected and disjointed edges
    edges_obj = domain.shapes[1]
    edges_tool = domain.shapes[1]
    (frag_edges, frag_vertices) = ShapeManipulation.intersect_edges(
        edges_obj,
        edges_tool,
        v_tag_shift=max_v_tag,
        e_tag_shift=max_e_tag,
        p_tag_shift=max_p_tag,
    )

    # append resulting fragments
    domain.append_shapes(frag_vertices)
    domain.append_shapes(frag_edges)

    # update wires
    domain.refresh_wires()
    domain.build_grahp()

    # Remove all shapes that are not presented in the graph
    domain.remove_vertex()
    domain.retag_shapes()
    domain.build_grahp()

    aka = 0

    return domain
