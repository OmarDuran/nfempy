import networkx as nx
import numpy as np
from numpy import linalg as la

import geometry.polygon_polygon_intersection_test as pp_intersector
import geometry.triangle_triangle_intersection_test as tt_intersector

from geometry.shape import Shape
from geometry.vertex import Vertex
from geometry.edge import Edge
from geometry.wire import Wire
from geometry.face import Face
from geometry.shell import Shell
from geometry.solid import Solid


class ShapeManipulation:
    def __init__(self, dimension):
        self.dimension = dimension

    @staticmethod
    def point_on_edge_boundary(point, edge):
        check = [np.all(np.isclose(point, vertex.point)) for vertex in edge.boundary_shapes]
        if check[0]:
            return edge.boundary_shapes[0]
        elif check[1]:
            return edge.boundary_shapes[1]
        else:
            return None

    @staticmethod
    def embed_vertex_in_edge(vertices: np.ndarray, edge: Edge, tag_shift=0):

        a: Vertex = edge.boundary_shapes[0]
        b: Vertex = edge.boundary_shapes[1]
        translation = a.point
        a.point = a.point - translation
        b.point = b.point - translation

        ra_norm = la.norm(a.point, axis=0)
        rb_norm = la.norm(b.point, axis=0)

        r_norms = [ra_norm, rb_norm]
        valid_indices = []
        for i, vertex in enumerate(vertices):
            obj_out = ShapeManipulation.point_on_edge_boundary(vertex.point, edge)
            if obj_out is not None:
                continue
            vertex.point = vertex.point - translation
            r_norm = la.norm(vertex.point, axis=0)
            if ra_norm < r_norm and r_norm < rb_norm:
                is_a = np.isclose(r_norm, ra_norm)
                is_b = np.isclose(r_norm, rb_norm)
                if not is_a and not is_b:
                    valid_indices.append(i)
                    r_norms.append(r_norm)
            vertex.point = vertex.point + translation
        a.point = a.point + translation
        b.point = b.point + translation
        if len(r_norms) > 2:
            perm = np.argsort(r_norms)
            vertex_list = [a, b] + list(vertices[valid_indices])
            vertex_list = [vertex_list[i] for i in perm]
            for i in range(len(vertex_list) - 1):
                e = Edge(i + tag_shift, np.array([vertex_list[i], vertex_list[i + 1]]))
                e.physical_tag = edge.physical_tag
                edge.immersed_shapes = np.append(edge.immersed_shapes, e)
        return vertices[valid_indices]

    @staticmethod
    def intersect_edges(edges_obj: np.ndarray, edges_tool: np.ndarray, axis=2, e_tag_shift=0, v_tag_shift=0, render_intersection_q = False):

        pos = np.array([0, 1])
        if axis == 0:
            pos = np.array([1, 2])
        elif axis == 1:
            pos = np.array([0, 2])

        vertices = np.array([], dtype=Shape)
        line_index = np.empty(shape=(0,2), dtype=int)
        edges = np.array([], dtype=Shape)
        # Collects intersection vertices

        tt_test = tt_intersector.TriangleTriangleIntersectionTest()

        v_tag = v_tag_shift
        e_tag = e_tag_shift

        case_idx = 0
        vertex_idx = 0
        vertex_map = {}
        obj_has_intersection_map = {}
        for i, edge_i in enumerate(edges_obj):
            if isinstance(edge_i, Wire):
                continue
            a, b = [shape.point for shape in edge_i.boundary_shapes]
            for j, edge_j in enumerate(edges_tool):
                if isinstance(edge_j, Wire):
                    continue
                p, q = [shape.point for shape in edge_j.boundary_shapes]
                if i >= j:
                    continue
                intersection_data = tt_test.line_line_intersection(
                    a, b, p, q, pos, render_intersection_q
                )
                if intersection_data[0]:
                    point = intersection_data[1]
                    line_index = np.vstack((line_index, np.array([i,j])))
                    v = Vertex(v_tag, point)
                    existence_check = np.array([np.all(np.isclose(v.point, vertex.point)) for vertex in vertices])
                    if len(existence_check) != 0 and np.all(existence_check):
                        v_index = np.argwhere(existence_check)[0,0]
                        vertex_map.__setitem__(case_idx, v_index)
                    else:
                        obj_out = ShapeManipulation.point_on_edge_boundary(point, edge_i)
                        tool_out = ShapeManipulation.point_on_edge_boundary(point, edge_j)
                        if obj_out is not None:
                            vertices = np.append(vertices, np.array([obj_out]), axis=0)
                        elif tool_out is not None:
                            vertices = np.append(vertices, np.array([tool_out]), axis=0)
                        else:
                            vertices = np.append(vertices, np.array([v]), axis=0)
                            v_tag += 1
                        vertex_map.__setitem__(case_idx, vertex_idx)
                        vertex_idx += 1

                    case_idx += 1

        for i, edge_i in enumerate(edges_obj):
            case_indices = np.argwhere(line_index == i)[:,0]
            v_indices = np.unique([vertex_map[i] for i in case_indices])
            local_vertices = ShapeManipulation.embed_vertex_in_edge(
                vertices[v_indices], edge_i, tag_shift=e_tag
            )
            if len(edge_i.immersed_shapes) > 0:
                new_e_tags = np.array([shape.tag for shape in edge_i.immersed_shapes])
                e_tag = np.max(new_e_tags) + 1
                assert len(local_vertices) == len(vertices[v_indices])
                edges = np.append(edges, edge_i.immersed_shapes)
        return (edges, vertices)

