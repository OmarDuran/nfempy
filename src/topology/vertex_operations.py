import numpy as np
from typing import Union
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell

from globals import topology_collapse_tol as collapse_tol
from topology.point_line_incidence import point_line_intersection


def vertex_with_same_geometry_q(
    vertex_a: Vertex, vertex_b: Vertex, eps: float = collapse_tol
):
    point_a = vertex_a.point
    point_b = vertex_b.point
    same_geometry_q = np.all(np.isclose(point_a, point_b, rtol=eps, atol=eps))
    return same_geometry_q


def vertex_strong_equality_q(vertex_a: Vertex, vertex_b: Vertex):
    shape_equality_q = vertex_a == vertex_b
    geometry_equality_q = vertex_with_same_geometry_q(vertex_a, vertex_b)
    strong_equality = shape_equality_q and geometry_equality_q
    return strong_equality

def collapse_vertex(
    v_tool: Vertex, v_object: Vertex, eps: float = collapse_tol
) -> Vertex:
    if vertex_strong_equality_q(v_tool, v_object):
        return v_object
    elif vertex_with_same_geometry_q(v_tool, v_object, eps):
        v_object.shape_assignment(v_tool)
        return v_object
    else:
        return None

def vertex_vertex_intersection(
    v_tool: Vertex, v_object: Vertex, eps: float = collapse_tol
) -> Union[None, Vertex]:
    if vertex_strong_equality_q(v_tool, v_object):
        return v_object
    elif vertex_with_same_geometry_q(v_tool, v_object, eps):
        v_object.shape_assignment(v_tool)
        return v_object
    else:
        return None

def vertex_edge_boundary_intersection(
    v_tool: Vertex, edge_object: Edge, eps: float = collapse_tol
) -> Union[None, Vertex]:
    check = [
        vertex_with_same_geometry_q(v_tool, e_vertex, eps)
        for e_vertex in edge_object.boundary_shapes
    ]
    if check[0]:
        return edge_object.boundary_shapes[0]
    elif check[1]:
        return edge_object.boundary_shapes[1]
    else:
        return None

def vertex_edge_intersection(
    v_tool: Vertex, edge_object: Edge, eps: float = collapse_tol
) -> Union[None, Vertex]:
    bc_out = vertex_edge_boundary_intersection(v_tool, edge_object, eps)
    if bc_out is not None:
        return bc_out

    p = v_tool.point
    a, b = edge_object.boundary_points()

    out = point_line_intersection(p, a, b)
    if out is None:
        return None
    else:
        tag = v_tool.tag
        return Vertex(tag, out)
