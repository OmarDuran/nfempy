from typing import Union
from topology.vertex import Vertex
from topology.edge import Edge

from globals import geometry_collapse_tol as collapse_tol
from globals import topology_tag_shape_info as tag_info
from globals import geometry_point_line_incidence_tol as p_incidence_tol
from globals import geometry_line_line_incidence_tol as l_incidence_tol
from topology.operations.vertex_operations import vertex_with_same_geometry_q
from topology.operations.vertex_operations import vertex_edge_boundary_intersection
from topology.operations.vertex_operations import vertex_edge_intersection
from geometry.operations.line_geometry_operations import line_line_intersection


def edge_with_same_geometry_q(edge_a: Edge, edge_b: Edge, eps: float = collapse_tol):
    va_0, va_1 = edge_a.boundary_shapes
    vb_0, vb_1 = edge_b.boundary_shapes
    v0_same_geometry_q = vertex_with_same_geometry_q(va_0, vb_0, eps)
    v1_same_geometry_q = vertex_with_same_geometry_q(va_1, vb_1, eps)
    same_geometry_q = v0_same_geometry_q and v1_same_geometry_q
    return same_geometry_q


def edge_strong_equality_q(edge_a: Edge, edge_b: Edge):
    shape_equality_q = edge_a == edge_b
    geometry_equality_q = edge_with_same_geometry_q(edge_a, edge_b)
    strong_equality = shape_equality_q and geometry_equality_q
    return strong_equality


def collapse_edge(e_tool: Edge, e_object: Edge, eps: float = collapse_tol) -> Vertex:
    if edge_strong_equality_q(e_tool, e_object):
        return e_object
    elif edge_with_same_geometry_q(e_tool, e_object, eps):
        e_object.shape_assignment(e_tool)
        return e_object
    else:
        return None


def edge_edge_boundary_intersection(
    e_tool: Edge, e_object: Edge, tag: int = tag_info.min, eps: float = p_incidence_tol
) -> Union[None, Vertex]:

    tv0, tv1 = e_tool.boundary_shapes
    # ov0, ov1 = e_object.boundary_shapes

    # intersection on boundary
    bc_out_tv0 = vertex_edge_boundary_intersection(tv0, e_object, tag, eps)
    bc_out_tv1 = vertex_edge_boundary_intersection(tv1, e_object, tag, eps)
    boundary_intersection_q = bc_out_tv0 is not None or bc_out_tv1 is not None

    if boundary_intersection_q:
        if bc_out_tv0 is not None:
            return bc_out_tv0
        else:
            return bc_out_tv1

    # intersection inside
    out_tv0 = vertex_edge_intersection(tv0, e_object, tag, eps)
    out_tv1 = vertex_edge_intersection(tv1, e_object, tag, eps)
    internal_intersection_q = out_tv0 is not None or out_tv1 is not None

    if internal_intersection_q:
        if out_tv0 is not None:
            return out_tv0
        else:
            return out_tv1

    if not boundary_intersection_q and not internal_intersection_q:
        return None


def edge_edge_intersection(
    e_tool: Edge,
    e_object: Edge,
    tag: int = tag_info.min,
    p_eps: float = p_incidence_tol,
    l_eps: float = l_incidence_tol,
) -> Union[None, Vertex]:
    bc_out = edge_edge_boundary_intersection(e_tool, e_object, p_eps)
    if bc_out is not None:
        return bc_out

    a, b = e_tool.boundary_points()
    c, d = e_object.boundary_points()
    out = line_line_intersection(a, b, c, d, l_eps)
    if out is None:
        return None
    else:
        return Vertex(tag, out)
