import numpy as np
from typing import Union, List, Tuple
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell

from globals import topology_tag_shape_info as tag_info
from globals import topology_collapse_tol as collapse_tol
from globals import topology_point_line_incidence_tol as p_incidence_tol
from globals import topology_point_line_incidence_tol as l_incidence_tol
from topology.point_line_incidence import point_line_incidence
from topology.point_line_incidence import point_line_intersection
from topology.point_line_incidence import points_line_intersection
from topology.point_line_incidence import points_line_argsort
from functools import partial

def vertex_with_same_geometry_q(
    vertex_a: Vertex, vertex_b: Vertex, eps: float = p_incidence_tol
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
    v_tool: Vertex, v_object: Vertex, tag: int = tag_info.min, eps: float = p_incidence_tol
) -> Union[None, Vertex]:
    if vertex_strong_equality_q(v_tool, v_object):
        return v_object  # same as v_tool
    elif vertex_with_same_geometry_q(v_tool, v_object, eps):
        vertex = Vertex(tag, v_tool.point)  # same geometry as v_tool
        return vertex
    else:
        return None


def vertex_edge_boundary_intersection(
    v_tool: Vertex, edge_object: Edge, tag: int = tag_info.min, eps: float = p_incidence_tol
) -> Union[None, Vertex]:
    strong_check = [
        vertex_strong_equality_q(v_tool, e_vertex)
        for e_vertex in edge_object.boundary_shapes
    ]
    weak_check = [
        vertex_with_same_geometry_q(v_tool, e_vertex, eps)
        for e_vertex in edge_object.boundary_shapes
    ]
    if strong_check[0]:
        return edge_object.boundary_shapes[0]
    elif strong_check[1]:
        return edge_object.boundary_shapes[1]
    if weak_check[0]:
        v0: Vertex = edge_object.boundary_shapes[0]
        vertex = Vertex(tag, v0.point)
        return vertex
    elif weak_check[1]:
        v1: Vertex = edge_object.boundary_shapes[1]
        vertex = Vertex(tag, v1.point)
        return vertex
    else:
        return None


def vertex_edge_intersection(
    v_tool: Vertex, edge_object: Edge, tag: int = tag_info.min, eps: float = p_incidence_tol
) -> Union[None, Vertex]:
    bc_out = vertex_edge_boundary_intersection(v_tool, edge_object, tag, eps)
    if bc_out is not None:
        return bc_out

    p = v_tool.point
    a, b = edge_object.boundary_points()

    out = point_line_intersection(p, a, b)
    if out is None:
        return None
    else:
        return Vertex(tag, out)

def vertices_edge_intersection(
    vertices_tool: Vertex, edge_object: Edge, tag: int = tag_info.min, eps: float = p_incidence_tol
) -> Tuple[np.ndarray, np.ndarray]:

    intersect_with_line = partial(vertex_edge_intersection, edge_object=edge_object,tag=tag_info.min,eps=eps)
    resulting_shapes = np.array(list(map(intersect_with_line,vertices_tool)))

    shapes_tool = []
    shape_intx = []
    for idx, shape in enumerate(resulting_shapes):
        if shape is None:
            continue
        shape.tag = tag
        tag += 1
        shape_intx.append(shape)
        shapes_tool.append(vertices_tool[idx])
    shape_intx = np.array(shape_intx)
    shapes_tool = np.array(shapes_tool)
    return shape_intx, shapes_tool

def vertices_edges_intersection(
    vertices_tool: Vertex, edges_object: Edge, tag: int = tag_info.min, eps: float = p_incidence_tol
) -> Tuple[np.ndarray, np.ndarray]:

    output = []
    for edge in  edges_object:
        shape_intx, shapes_tool = vertices_edge_intersection(vertices_tool, edge, tag, eps)
        tag = np.max([shape.tag for shape in shape_intx])
        output.append((edge, shape_intx, shapes_tool))
    return output

def vertices_edge_difference(vertices_tool: Vertex, edge_object: Edge, tag: int = tag_info.min, eps: float = l_incidence_tol):

    # deduplication of vertices
    points_tool = np.array([vertex.point for vertex in vertices_tool])

    precision = 12
    points_rounded = np.round(points_tool, decimals=precision)
    _, idx = np.unique(points_rounded, axis=0, return_index=True)
    points_tool = points_tool[idx]

    a, b = edge_object.boundary_points()
    vertex_a, vertex_b = edge_object.boundary_shapes
    # filter no incident points
    points, intx_q = points_line_intersection(points_tool, a, b, eps)
    if len(points) == 0 and np.all(np.logical_not(intx_q)): # no new shapes are generated
        return [], []

    idx = points_line_argsort(points, a, b)
    internal_idx = idx.copy()

    first_point_is_on_a = np.all(np.isclose(points[idx][0], a))
    last_point_is_on_b = np.all(np.isclose(points[idx][-1], b))
    if first_point_is_on_a:
        internal_idx = np.delete(internal_idx, 0)
    if last_point_is_on_b:
        internal_idx = np.delete(internal_idx, -1)

    # including end points
    expanded_pts = points[internal_idx]
    expanded_pts = np.insert(expanded_pts, 0, a, axis=0)
    expanded_pts = np.append(expanded_pts, np.array([b]), axis = 0)

    ridx = np.arange(expanded_pts.shape[0])

    # Create a point chain
    chains = np.array([(expanded_pts[ridx[i]], expanded_pts[ridx[i+1]]) for i in range(len(ridx) - 1)])
    new_vertices = []
    new_edges = []
    for chain in chains:
        if vertex_with_same_geometry_q(Vertex(tag, chain[0]), vertex_a):
            if first_point_is_on_a:
                v0 = vertices_tool[intx_q][idx][0]
            else:
                v0 = vertex_a
        else:
            v0 = Vertex(tag, chain[0])
            v0.physical_tag = edge_object.physical_tag
            tag += 1
        if vertex_with_same_geometry_q(Vertex(tag, chain[1]), vertex_b):
            if last_point_is_on_b:
                v1 = vertices_tool[intx_q][idx][-1]
            else:
                v1 = vertex_b
        else:
            v1 = Vertex(tag, chain[1])
            v1.physical_tag = edge_object.physical_tag
            tag += 1
        e = Edge(tag, np.array([v0, v1]))
        e.physical_tag = edge_object.physical_tag
        tag += 1
        new_vertices.append(v0)
        new_vertices.append(v1)
        new_edges.append(e)

    return new_edges, new_vertices


def vertices_edges_difference(vertices_tool: Vertex, edges_object: Edge, tag: int = tag_info.min, eps: float = l_incidence_tol):
    output = []
    for edge in  edges_object:
        new_edges, new_vertices = vertices_edge_difference(vertices_tool, edge, tag, eps)
        if len(new_vertices) != 0 and len(new_edges) != 0:
            tag = np.max([shape.tag for shape in new_vertices] + [shape.tag for shape in new_edges]) + 1
        output.append((edge, new_edges, new_vertices))
    return output