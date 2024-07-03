import numpy as np
from topology.domain import Domain
from globals import topology_point_line_incidence_tol as l_incidence_tol
from globals import topology_tag_shape_info as tag_info
from topology.vertex import Vertex
from topology.edge import Edge

from topology.point_line_incidence import points_line_intersection
from topology.point_line_incidence import points_line_argsort
from topology.vertex_operations import vertex_with_same_geometry_q
from topology.vertex_operations import vertices_edge_intersection
from topology.vertex_operations import vertex_edge_intersection

from mesh.conformal_mesher import ConformalMesher
from mesh.mesh import Mesh

def create_conformal_mesher(domain: Domain, h, ref_l=0):
    mesher = ConformalMesher(dimension=domain.dimension)
    mesher.domain = domain
    mesher.generate_from_domain(h, ref_l)
    mesher.write_mesh("gmesh.msh")
    return mesher

def index_shape_by_co_dimension(shape, max_dim: int = 1):
    return (max_dim - shape.dimension, shape.tag)

def append_shape(shapes, shape, max_dim: int = 1):
    co_dim = max_dim - shape.dimension
    shapes[co_dim][shape.hash()] = shape
    return

# case one "fractured line"
v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
v1: Vertex = Vertex(1, np.array([1.0, 1.0, 1.0]))
e0: Edge = Edge(0, np.array([v0, v1]))

v2: Vertex = Vertex(2, np.array([0.25, 0.25, 0.25]))
v3: Vertex = Vertex(3, np.array([0.5, 0.5, 0.5]))



# All co_dimensions are persistent
shapes = [{}, {}]
[append_shape(shapes, shape) for shape in [v0, v1, v2, v3, e0]]

# Step 1: Compute intersection
vertices_tool = np.array([v2, v3])
intxs_e0, vertices_tool = vertices_edge_intersection(vertices_tool, e0, tag=4)
[append_shape(shapes, shape) for shape in intxs_e0]

# Step 2: Compute difference
def vertices_edge_difference(vertices_tool: Vertex, edge_object: Edge, tag: int = tag_info.min, eps: float = l_incidence_tol):

    # deduplication of vertices
    points_tool = np.array([vertex.point for vertex in vertices_tool])
    points_tool, v_indices = np.unique(points_tool, return_index=True, axis=0)
    vertices_tool = vertices_tool[v_indices]

    a, b = edge_object.boundary_points()
    vertex_a, vertex_b = edge_object.boundary_shapes
    # filter no incident points
    points = points_line_intersection(points_tool, a, b, eps)
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
            v0 = vertex_a
        else:
            v0 = Vertex(tag, chain[0])
            tag += 1
        if vertex_with_same_geometry_q(Vertex(tag, chain[1]), vertex_b):
            v1 = vertex_b
        else:
            v1 = Vertex(tag, chain[1])
            tag += 1
        e = Edge(tag, np.array([v0,v1]))
        tag += 1
        new_vertices.append(v0)
        new_vertices.append(v1)
        new_edges.append(e)

    return new_vertices, new_edges
parts = vertices_edge_difference(intxs_e0, e0, tag=7)

domain = Domain(dimension=1)
domain.append_shapes(parts[0])
domain.append_shapes(parts[1])
domain.build_grahp()
domain.draw_grahp()

h_val = 0.1
mesher = create_conformal_mesher(domain, h_val, 0)

aka = 0
