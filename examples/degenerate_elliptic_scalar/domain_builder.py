import numpy as np

from topology.domain import Domain
from topology.edge import Edge
from topology.face import Face
from topology.vertex import Vertex
from topology.wire import Wire


def build_line_1D(line_points, physical_tags=None):
    if physical_tags is None:
        physical_tags = {"line_0": 1, "line_1": 2, "bc_0": 3, "bc_1": 4}

    domain = Domain(dimension=1)
    vertices = np.array([Vertex(tag, point) for tag, point in enumerate(line_points)])
    domain.append_shapes(vertices)

    domain.shapes[0][0].physical_tag = physical_tags.get("bc_0", None)
    domain.shapes[0][2].physical_tag = physical_tags.get("bc_1", None)

    shape_tag = 0
    for idx in range(len(vertices) - 1):
        vertex_indices = [idx, idx + 1]
        edge = Edge(shape_tag, domain.shapes[0][vertex_indices])
        edge.physical_tag = physical_tags.get("line_" + str(shape_tag), None)
        domain.append_shapes(np.array([edge]))
        shape_tag += 1

    domain.build_grahp()
    return domain


def build_surface_2D(surface_points, physical_tags=None):
    if physical_tags is None:
        physical_tags = {"area_0": 1, "area_1": 2}

    domain = Domain(dimension=2)
    vertices = np.array(
        [Vertex(tag, point) for tag, point in enumerate(surface_points)]
    )
    domain.append_shapes(vertices)

    edges_connectivities = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 0],
        [4, 6],
        [6, 2],
    ]
    edges_list = []
    e_physical_tag = 3
    skip_e_tags = [6, 7]
    for e_tag, edge_con in enumerate(edges_connectivities):
        edge = Edge(e_tag, domain.shapes[0][edge_con])
        if e_tag not in skip_e_tags:
            edge.physical_tag = e_physical_tag
            e_physical_tag += 1
        edges_list.append(edge)

    edges = np.array(edges_list)
    domain.append_shapes(edges)

    edge_idxs = np.array([2, 3, 6, 7])
    vertices = np.array([domain.shapes[0][2], domain.shapes[0][2]])
    e_tag += 1
    wire = Wire(e_tag, edges[edge_idxs], vertices)
    domain.append_shapes(np.array([wire]))
    surface = Face(0, np.array([wire]))
    surface.physical_tag = physical_tags.get("area_0", None)
    domain.append_shapes(np.array([surface]))

    edge_idxs = np.array([0, 1, 7, 6, 4, 5])
    vertices = np.array([domain.shapes[0][0], domain.shapes[0][0]])
    e_tag += 1
    wire = Wire(e_tag, edges[edge_idxs], vertices)
    domain.append_shapes(np.array([wire]))
    surface = Face(1, np.array([wire]))
    surface.physical_tag = physical_tags.get("area_1", None)
    domain.append_shapes(np.array([surface]))

    domain.build_grahp()
    return domain
