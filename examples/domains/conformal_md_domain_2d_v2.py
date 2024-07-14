

import numpy as np
from globals import topology_tag_shape_info
from globals import geometry_collapse_precision as collapse_precision
from topology.domain_market import build_box_2D
from topology.operations.domain_operations import create_domain
from topology.operations.domain_operations import domain_difference
from topology.operations.domain_operations import domain_union
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face


from geometry.operations.point_geometry_operations import points_line_intersection
from geometry.operations.point_geometry_operations import points_line_argsort
from geometry.operations.point_geometry_operations import points_polygon_intersection
from geometry.operations.line_geometry_operations import lines_lines_intersection
from mesh.discrete_domain import DiscreteDomain
from mesh.mesh import Mesh

# input
# define a disjoint fracture geometry
points = np.array([
    [0.0, 0.5, 0.0],
    [1.0, 0.5, 0.0],
    [0.5, 0.0, 0.0],
    [0.5, 1.0, 0.0],
    [0.75, -0.2, 0.0],
    [0.75, 1.2, 0.0],
    [0.0, 0.0, 0.0],
    [1.2, 1.2, 0.0],
    [-0.2, 0.75, 0.0],
    [1.2, 0.75, 0.0],
    [0.1, 0.25, 0.0],
    [0.9, 0.25, 0.0],
])
frac_idx = np.array([
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
])
lines = points[frac_idx]

lx = 1.0
ly = 1.0
domain_physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}
fracture_physical_tags = {"line": 10, "point": 20, "internal_bc": 15}
box_points = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0]])

# processing
rock_domain = build_box_2D(box_points, domain_physical_tags)
rock_domain_vertices = [shape for shape in rock_domain.shapes[0]]
rock_domain_edges = [shape for shape in rock_domain.shapes[1] if not shape.composite]
rock_domain_lines = np.array([shape.boundary_points() for shape in rock_domain_edges])
boundary_intx = lines_lines_intersection(lines_tools=lines,lines_objects=rock_domain_lines, deduplicate_points_q=True)
fracture_intx = lines_lines_intersection(lines_tools=lines,lines_objects=lines, deduplicate_points_q=True)

boundary_vertices = {}
for i, line in enumerate(rock_domain_lines):
    a,b = line
    out, intx_idx = points_line_intersection(boundary_intx, a, b)
    if len(out) == 0:
        continue
    boundary_vertices[i] = out

raw_vertices = []
for item in boundary_vertices.items():
    i, intx_points = item
    for point in intx_points:
        v: Vertex = Vertex(topology_tag_shape_info.min, point)
        v.physical_tag = rock_domain_edges[i].physical_tag
        raw_vertices.append(v)

# Eliminate duplicates values since face boundaries can not share a vertex (line boundary)
b_points = np.array([vertex.point for vertex in raw_vertices])
b_points_rounded = np.round(b_points, decimals=collapse_precision)
_, unique_idx = np.unique(b_points_rounded, axis=0, return_index=True)

tag = rock_domain.max_tag() + 1
vertices = []
for i, vertex in enumerate(raw_vertices):
    if i in unique_idx:
        vertex.tag = tag
        vertices.append(vertex)
        tag += 1

for point in fracture_intx:
    v: Vertex = Vertex(tag, point)
    v.physical_tag = fracture_physical_tags['point']
    vertices.append(v)
    tag += 1

domain_c1 = create_domain(dimension=1,  shapes=np.array(vertices))
# domain_c1.build_grahp(0)
# domain_c1.draw_grahp()

edges = []
for line in lines:
    edge_bc = []
    for point in line:
        v: Vertex = Vertex(tag, point)
        v.physical_tag = fracture_physical_tags['internal_bc']
        vertices.append(v)
        edge_bc.append(v)
        tag += 1
    e: Edge = Edge(tag, np.array(edge_bc))
    e.physical_tag = fracture_physical_tags['line']
    edges.append(e)
    tag += 1

vertices = np.array(vertices)
edges = np.array(edges)
domain = create_domain(dimension=1, shapes=[])
domain.append_shapes(rock_domain_vertices)
domain.append_shapes(vertices)
domain.append_shapes(rock_domain_edges)
domain.append_shapes(edges)
# domain.build_grahp()

domain_c0 = domain_difference(domain, domain_c1, tag)
md_domain_c1 = domain_union(domain_c0, domain_c1)

# remove shapes outside the original domain
vertices = np.array([vertex for vertex in md_domain_c1.shapes[0]])
edges = np.array([edge for edge in md_domain_c1.shapes[1]])
points = np.array([vertex.point for vertex in vertices])
v_out, v_intx_q = points_polygon_intersection(points,box_points)
edges_xcs = np.array([np.mean(edge.boundary_points(),axis = 0) for edge in edges])
e_out, e_intx_q = points_polygon_intersection(edges_xcs,box_points)

vertices = vertices[v_intx_q]
edges = edges[e_intx_q]

wire_edges = []
for boundary_edge in rock_domain_edges:
    sub_edges = np.array([edge for edge in edges if edge.physical_tag == boundary_edge.physical_tag])
    sub_edges_xcs = np.array([np.mean(sub_edge.boundary_points(), axis=0) for sub_edge in sub_edges])
    a, b = boundary_edge.boundary_points()
    idx = points_line_argsort(sub_edges_xcs, a, b)
    wire_edges.append(sub_edges[idx])
wire_edges = np.concatenate(wire_edges)
wire_boundary = wire_edges[0].boundary_shapes[[0]] # closed
wire0: Wire = Wire(tag, wire_edges, wire_boundary)
tag += 1
face0: Face = Face(tag, np.array([wire0]))
face0.immersed_shapes = edges
face0.physical_tag = domain_physical_tags['area']
tag += 1

md_domain = create_domain(dimension=2, shapes=[])
md_domain.append_shapes(vertices)
md_domain.append_shapes(edges)
md_domain.append_shapes(np.array([wire0]))
md_domain.append_shapes(np.array([face0]))


# Conformal gmsh discrete representation
h_val = 0.1
domain_h = DiscreteDomain(dimension=md_domain.dimension)
domain_h.domain = md_domain
domain_h.generate_mesh(h_val, 0)
domain_h.write_mesh("gmesh.msh")

# Mesh representation
gmesh = Mesh(dimension=md_domain.dimension, file_name="gmesh.msh")
gmesh.build_conformal_mesh()
gmesh.write_vtk()

