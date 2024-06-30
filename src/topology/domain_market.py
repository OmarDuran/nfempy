import csv

import numpy as np

from topology.domain import Domain
from topology.edge import Edge
from topology.face import Face
from topology.shape_manipulation import embed_edge_in_face
from topology.shell import Shell
from topology.solid import Solid
from topology.vertex import Vertex
from topology.wire import Wire


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
    domain.build_grahp()
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
    domain.build_grahp()
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
        domain.append_shapes(np.array([edge]))
    tag += 1

    wire_0 = Wire(tag, domain.shapes[1][[0, 1, 2, 3]], domain.shapes[0][[0]])
    tag += 1

    wire_1 = Wire(tag, domain.shapes[1][[4, 5, 6, 7]], domain.shapes[0][[4]])
    tag += 1

    wire_2 = Wire(tag, domain.shapes[1][[0, 9, 4, 8]], domain.shapes[0][[0]])
    tag += 1

    wire_3 = Wire(tag, domain.shapes[1][[1, 10, 5, 9]], domain.shapes[0][[1]])
    tag += 1

    wire_4 = Wire(tag, domain.shapes[1][[2, 11, 6, 10]], domain.shapes[0][[2]])
    tag += 1

    wire_5 = Wire(tag, domain.shapes[1][[3, 11, 7, 8]], domain.shapes[0][[3]])
    tag += 1

    wires = np.array([wire_0, wire_1, wire_2, wire_3, wire_4, wire_5])
    domain.append_shapes(wires)

    bc_labels = ["bc_0", "bc_1", "bc_2", "bc_3", "bc_4", "bc_5"]
    for bc_label, wire in zip(bc_labels, wires):
        surface = Face(tag, np.array([wire]))
        surface.physical_tag = physical_tags.get(bc_label, None)
        domain.append_shapes(np.array([surface]))
        tag += 1

    shell = Shell(
        tag,
        domain.shapes[2],
        wires,
    )
    domain.append_shapes(np.array([shell]))
    tag += 1

    solid = Solid(tag, np.array([shell]))
    solid.physical_tag = physical_tags.get("solid", None)
    domain.append_shapes(np.array([solid]))
    domain.build_grahp()
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

    embed_edge_in_face(domain_lines.shapes[1], face)
    domain.append_shapes(domain_lines.shapes[0])
    domain.append_shapes(domain_lines.shapes[1])

    max_v_tag = len(domain.shapes[0])
    max_e_tag = len(domain.shapes[1])
    max_p_tag = domain.max_physical_tag()

    # performing multiple intersection of connected and disjointed edges
    edges_obj = domain.shapes[1]
    edges_tool = domain.shapes[1]
    (frag_edges, frag_vertices) = intersect_edges(
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

    return domain


def build_disjoint_planes(
    file_name, max_f_tag=0, max_e_tag=0, max_v_tag=0, max_p_tag=0
):
    domain = Domain(dimension=3)
    planes = read_fractures_file(4, file_name)
    physical_tags = [i + max_p_tag for i in range(len(planes))]
    v_tag = max_v_tag
    e_tag = max_e_tag
    f_tag = max_f_tag
    for plane, physical_tag in zip(planes, physical_tags):
        v0 = Vertex(v_tag, plane[0])
        v_tag += 1
        v1 = Vertex(v_tag, plane[1])
        v_tag += 1
        v2 = Vertex(v_tag, plane[2])
        v_tag += 1
        v3 = Vertex(v_tag, plane[3])
        v_tag += 1

        e0 = Edge(e_tag, np.array([v0, v1]))
        e_tag += 1

        e1 = Edge(e_tag, np.array([v1, v2]))
        e_tag += 1

        e2 = Edge(e_tag, np.array([v2, v3]))
        e_tag += 1

        e3 = Edge(e_tag, np.array([v3, v0]))
        e_tag += 1

        w0 = Wire(e_tag, np.array([e0, e1, e2, e3]), np.array([v0]))
        e_tag += 1

        s0 = Face(f_tag, np.array([w0]))
        s0.physical_tag = physical_tag
        f_tag += 1
        physical_tag += 1

        domain.append_shapes(np.array([v0, v1, v2, v3]))
        domain.append_shapes(np.array([e0, e1, e2, e3, w0]))
        domain.append_shapes(np.array([s0]))

    return domain


def build_box_3D_with_planes(box_points, planes_file, physical_tags=None):
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
    domain = build_box_3D(box_points, physical_tags)
    max_v_tag = len(domain.shapes[0])
    max_e_tag = len(domain.shapes[1])
    max_f_tag = len(domain.shapes[2])
    max_p_tag = domain.max_physical_tag()
    solid = domain.shapes[3][0]

    # planes
    domain_planes = build_disjoint_planes(
        planes_file,
        max_f_tag=max_f_tag,
        max_e_tag=max_e_tag,
        max_v_tag=max_v_tag,
        max_p_tag=max_p_tag + 1,
    )

    # step one embed faces in solid
    faces = domain_planes.shapes[2]
    solid.immersed_shapes = np.append(solid.immersed_shapes, np.array([faces]))
    domain.append_shapes(domain_planes.shapes[0])
    domain.append_shapes(domain_planes.shapes[1])
    domain.append_shapes(domain_planes.shapes[2])

    max_v_tag = len(domain.shapes[0])
    max_e_tag = len(domain.shapes[1])
    max_f_tag = len(domain.shapes[2])
    max_p_tag = domain.max_physical_tag()

    # step two multiple plane intersection of of connected and disjointed planes
    faces_obj = domain.shapes[2]
    faces_tool = domain.shapes[2]
    (frag_edges, frag_vertices) = intersect_faces(
        faces_obj,
        faces_tool,
        f_tag_shift=max_f_tag,
        v_tag_shift=max_v_tag,
        e_tag_shift=max_e_tag,
        p_tag_shift=max_p_tag,
    )

    # step 3
    # append resulting fragments
    domain.append_shapes(frag_vertices)
    domain.append_shapes(frag_edges)

    # step 4 update wires and shells
    # update wires
    domain.refresh_wires()
    domain.build_grahp()

    # Remove all shapes that are not presented in the graph
    # domain.remove_vertex()
    # domain.retag_shapes()
    # domain.build_grahp()

    return domain
