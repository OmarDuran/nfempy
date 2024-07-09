
import numpy as np
import networkx as nx

from topology.domain import Domain
from topology.domain_operations import create_domain, domain_intersection

from globals import topology_tag_shape_info as tag_info
from topology.vertex import Vertex
from topology.edge import Edge

from topology.point_line_incidence import points_line_intersection
from topology.point_line_incidence import point_line_intersection
from topology.point_line_incidence import points_line_argsort
from topology.vertex_operations import vertex_with_same_geometry_q
from topology.vertex_operations import vertices_edge_intersection
from topology.vertex_operations import vertices_edge_difference

from mesh.discrete_domain import DiscreteDomain
from mesh.mesh import Mesh

def plot_graph(G):
    nx.draw(
        G,
        pos=nx.circular_layout(G),
        with_labels=True,
        node_color="skyblue",
    )

def append_shape(shapes, shape, max_dim: int = 1):
    co_dim = max_dim - shape.dimension
    shapes[co_dim][shape.index(max_dimension=max_dim)] = shape
    return

max_dim = 1
# shape collection
shapes = [{}, {}]

# case one "fractured line"
# Domain definition
v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
v1: Vertex = Vertex(1, np.array([1.0, 1.0, 1.0]))
e0: Edge = Edge(0, np.array([v0, v1]))
v0.physical_tag = 2 # physical tag for BC of pdes on c0
v1.physical_tag = 3 # physical tag for BC of pdes on c0
e0.physical_tag = 1 # physical tag for pdes on c0

domain = create_domain(dimension=max_dim, shapes=np.array([v0, v1, e0]))

# Define domains to be subtracted
v2: Vertex = Vertex(2, np.array([0.25, 0.25, 0.25]))
v2.physical_tag = 10 # physical tag for pdes on c1

c1_domain = create_domain(dimension=max_dim, shapes=np.array([v2]))

intx_domain = domain_intersection(c1_domain, domain)

[append_shape(shapes, shape) for shape in [v0, v1, v2, e0]]

# Step 1: Compute intersection
vertices_tool = np.array([v2])
intxs_e0, vertices_tool = vertices_edge_intersection(vertices_tool, e0, tag=4)
[append_shape(shapes, shape) for shape in intxs_e0]


# Step 2: Compute difference
parts = vertices_edge_difference(intxs_e0, e0, tag=7)
[append_shape(shapes, shape) for shape in parts[0]]
[append_shape(shapes, shape) for shape in parts[1]]

# shapes to embed
v10: Vertex = Vertex(20, np.array([0.125, 0.125, 0.125]))
v11: Vertex = Vertex(21, np.array([0.75, 0.75, 0.75]))
v10.physical_tag = 20
v11.physical_tag = 20
vertices_to_embed = np.array([v10, v11])
# check for shapes that can be embedded
def is_vertex_embedded_in_edge(vertex: Vertex, edge: Edge):
    p = vertex.point
    a, b = edge.boundary_points()
    out = point_line_intersection(p,a,b)
    if out is None:
        return False
    else:
        return True
vertices_embeddability_q= []
for vertex, edge in zip(vertices_to_embed,parts[0]):
    vertices_embeddability_q.append(is_vertex_embedded_in_edge(vertex,edge))

for i, check in enumerate(vertices_embeddability_q):
    if check:
        parts[0][i].immersed_shapes = vertices_to_embed[[i]]

domain_broken = Domain(dimension=1)
domain_broken.append_shapes(parts[0])
domain_broken.append_shapes(parts[1])
domain_broken.append_shapes(vertices_to_embed[vertices_embeddability_q])
domain_broken.build_grahp()
# domain_broken.draw_grahp()

md_G_partial = domain_broken.graph.copy()
# Perform union
vertex_interfaces = {}
for v_tool in vertices_tool:
    v_tool_index = v_tool.index(max_dimension=max_dim)
    chunk = [vertex.index(max_dimension=max_dim) for vertex in parts[1] if vertex_with_same_geometry_q(vertex, v_tool)]
    vertex_interfaces[v_tool.hash()] = chunk
    predecessors = [list(md_G_partial.predecessors(index))[0] for index in chunk]
    [md_G_partial.add_edge(index,v_tool_index) for index in predecessors]
    md_G_partial.remove_nodes_from(chunk)


# plot_graph(md_G_partial)
# Step 1: remove embeddings from the graph. This is a BRep graph
nodes_to_remove = [shape.index(max_dimension=max_dim) for shape in vertices_to_embed[vertices_embeddability_q]]
md_G_partial.remove_nodes_from(nodes_to_remove)
# plot_graph(md_G_partial)

# Step 2: convert the BRep-graph to a BRep-domain
indexed_shapes = md_G_partial.nodes()
c0_indexed_shapes = [index for index in indexed_shapes if index[0] == 0]
c1_indexed_shapes = [index for index in indexed_shapes if index[0] == 1]

c0_shapes = np.array([shapes[0][index] for index in c0_indexed_shapes])
c1_shapes = np.array([shapes[1][index] for index in c1_indexed_shapes])


# for all c0 shapes
max_co_dim = 1
for shape_idx, c0_indexed_shape in enumerate(c0_indexed_shapes):
    c0_shape = c0_shapes[shape_idx]
    shape_successors = nx.dfs_successors(md_G_partial, source=c0_indexed_shape, depth_limit=max_co_dim)
    brep_shape_index = shape_successors[c0_indexed_shape]
    brep_shapes = np.array([shapes[1][index] for index in brep_shape_index])
    # swap_brep_shape
    c0_shape.boundary_shapes = brep_shapes


md_brep_domain = Domain(dimension=1)
md_brep_domain.append_shapes(c0_shapes)
md_brep_domain.append_shapes(c1_shapes)
md_brep_domain.append_shapes(vertices_to_embed[vertices_embeddability_q])
md_brep_domain.build_grahp()
md_brep_domain.draw_grahp()

# Conformal gmsh discrete representation
h_val = 0.1
domain_h = DiscreteDomain(dimension=md_brep_domain.dimension)
domain_h.domain = md_brep_domain
domain_h.generate_mesh(h_val, 0)
domain_h.write_mesh("gmesh.msh")

# Mesh representation
gmesh = Mesh(dimension=md_brep_domain.dimension, file_name="gmesh.msh")
gmesh.build_conformal_mesh()
gmesh.write_vtk()


aka = 0
