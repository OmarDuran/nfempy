import numpy as np
import networkx as nx
import copy
from topology.domain import Domain


# supported shape operations
from topology.vertex_operations import vertices_edges_intersection
from topology.vertex_operations import vertices_edges_difference
from topology.vertex_operations import vertex_with_same_geometry_q

from typing import Union, List, Tuple
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face
from topology.shell import Shell

from globals import topology_tag_shape_info as tag_info
from globals import topology_point_line_incidence_tol as p_incidence_tol

def __append_shape(shapes, shape, max_dim: int = 1):
    co_dim = max_dim - shape.dimension
    shapes[co_dim][shape.index(max_dimension=max_dim)] = shape
    return

def __plot_graph(G):
    nx.draw(
        G,
        pos=nx.circular_layout(G),
        with_labels=True,
        node_color="skyblue",
    )

def create_domain(dimension, shapes: np.array):
    domain = Domain(dimension=dimension)

    # classify shapes
    shapes_by_co_dimension = [{} for _ in range(dimension + 1)]
    for shape in shapes:
        co_dimension = dimension - shape.dimension
        shapes_by_co_dimension[co_dimension][shape.hash()] = shape

    # append shapes
    for c_shapes in shapes_by_co_dimension:
        shape_list = list(c_shapes.values())
        domain.append_shapes(np.array(shape_list))
    return domain

def domain_intersection(c1_tool:Domain, c0_object:Domain, tag: int = tag_info.min, eps: float = p_incidence_tol):

    max_dim = np.max([c1_tool.dimension, c0_object.dimension])

    c0_shapes_object = [shape for dim_shape in c0_object.shapes for shape in dim_shape if max_dim - shape.dimension == 0]
    c1_shapes_tool = [shape for dim_shape in c1_tool.shapes for shape in dim_shape if max_dim - shape.dimension == 1]

    # case disjoint vertices and disjoint edges
    e_object = np.array(c0_shapes_object)
    v_tool = np.array(c1_shapes_tool)
    output = vertices_edges_intersection(v_tool, e_object, tag, eps)

    # collect c1 shapes resulting from the intersection
    new_c1_shapes = []
    for edge, shape_intx, shape_tool in output:
        new_c1_shapes.append(shape_intx)

    if len(new_c1_shapes) == 0:
        return None
    else:
        new_c1_shapes = np.concatenate(new_c1_shapes)
        intx_domain = create_domain(dimension=max_dim, shapes=new_c1_shapes)
        return intx_domain

def domain_difference(domain_minuend:Domain, domain_subtrahend:Domain, tag: int = tag_info.min, eps: float = p_incidence_tol):

    max_dim = np.max([domain_minuend.dimension, domain_subtrahend.dimension])

    c0_shapes_object = [shape for dim_shape in domain_minuend.shapes for shape in dim_shape if max_dim - shape.dimension == 0]
    c1_shapes_tool = [shape for dim_shape in domain_subtrahend.shapes for shape in dim_shape if max_dim - shape.dimension == 1]

    # case: disjoint vertices and disjoint edges
    e_object = np.array(c0_shapes_object)
    v_tool = np.array(c1_shapes_tool)
    output = vertices_edges_difference(v_tool, e_object, tag, eps)

    assert len(e_object) == len(output)

    shapes_by_co_dimension = [{} for _ in range(max_dim + 1)]
    for edge, shape_edges, shape_vertices in output:
        for shape in shape_edges:
            co_dimension = max_dim - shape.dimension
            shapes_by_co_dimension[co_dimension][shape.index(max_dimension=max_dim)] = shape
        for shape in shape_vertices:
            co_dimension = max_dim - shape.dimension
            shapes_by_co_dimension[co_dimension][shape.index(max_dimension=max_dim)] = shape

        # case for no intersections
        if len(shape_edges) == 0 and len(shape_vertices) == 0:
            co_dimension = max_dim - edge.dimension
            shapes_by_co_dimension[co_dimension][edge.index(max_dimension=max_dim)] = edge
            for bc_shape in edge.boundary_shapes:
                co_dimension = max_dim - bc_shape.dimension
                shapes_by_co_dimension[co_dimension][
                    bc_shape.index(max_dimension=max_dim)] = bc_shape

    # Try to create the domain
    diff_domain = Domain(dimension=max_dim)
    for c_shapes in shapes_by_co_dimension:
        shape_list = list(c_shapes.values())
        diff_domain.append_shapes(np.array(shape_list))
    if len(diff_domain.shapes) == 0:
        return None
    else:
        return diff_domain

def domain_union(domain_c0:Domain, domain_c1:Domain, tag: int = tag_info.min, eps: float = p_incidence_tol):

    max_dim = np.max([domain_c0.dimension, domain_c1.dimension])
    all_shapes = [{} for _ in range(max_dim + 1)]
    for shape_by_dimension in domain_c0.shapes:
        for shape in shape_by_dimension:
            co_dimension = max_dim - shape.dimension
            all_shapes[co_dimension][shape.index(max_dimension=max_dim)] = copy.copy(shape)
    for shape_by_dimension in domain_c1.shapes:
        for shape in shape_by_dimension:
            co_dimension = max_dim - shape.dimension
            all_shapes[co_dimension][shape.index(max_dimension=max_dim)] = copy.copy(shape)

    domain_c0.build_grahp()
    md_G_partial = domain_c0.graph.copy()

    # Perform union by including c1 domains into c0 domain
    # c1_interfaces = {}
    for v_tool in domain_c1.shapes[0]:
        v_tool_index = v_tool.index(max_dimension=max_dim)
        chunk = [vertex.index(max_dimension=max_dim) for vertex in domain_c0.shapes[0] if
                 vertex_with_same_geometry_q(vertex, v_tool)]
        # filter v_tool if is included already in the graph
        chunk = [index for index in chunk if index != v_tool_index]
        predecessors = [list(md_G_partial.predecessors(index))[0] for index in chunk]
        [md_G_partial.add_edge(index, v_tool_index) for index in predecessors]
        md_G_partial.remove_nodes_from(chunk)

    # Step 1: remove embeddings from the graph. This is a BRep graph
    # Step 2: convert the BRep-graph to a BRep-domain
    indexed_shapes = md_G_partial.nodes()
    c0_indexed_shapes = [index for index in indexed_shapes if index[0] == 0]
    c1_indexed_shapes = [index for index in indexed_shapes if index[0] == 1]

    c0_shapes = np.array([all_shapes[0][index] for index in c0_indexed_shapes])
    c1_shapes = np.array([all_shapes[1][index] for index in c1_indexed_shapes])

    # for all c0 shapes swap brep
    for shape_idx, c0_indexed_shape in enumerate(c0_indexed_shapes):
        c0_shape = c0_shapes[shape_idx]
        shape_successors = nx.dfs_successors(md_G_partial, source=c0_indexed_shape, depth_limit=max_dim)
        brep_shape_index = shape_successors[c0_indexed_shape]
        brep_shapes = np.array([all_shapes[1][index] for index in brep_shape_index])
        # swap_brep_shape
        c0_shape.boundary_shapes = brep_shapes

    # Try to create the domain
    md_brep_shapes = np.concatenate([c1_shapes,c0_shapes])
    md_brep_domain = create_domain(dimension=max_dim, shapes=md_brep_shapes)
    if len(md_brep_domain.shapes) == 0:
        return None
    else:
        return md_brep_domain