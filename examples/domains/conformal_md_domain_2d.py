

import numpy as np
from topology.operations.domain_operations import create_domain
from topology.operations.domain_operations import domain_difference
from topology.operations.domain_operations import domain_union
from topology.vertex import Vertex
from topology.edge import Edge
from topology.wire import Wire
from topology.face import Face

from topology.line_line_incidence import lines_lines_intersection

from mesh.discrete_domain import DiscreteDomain
from mesh.mesh import Mesh

max_dim = 1
# case one "fractured surface"

# Domain definition

v0: Vertex = Vertex(0, np.array([0.0, 0.0, 0.0]))
v1: Vertex = Vertex(1, np.array([1.0, 0.0, 0.0]))
v2: Vertex = Vertex(2, np.array([1.0, 1.0, 0.0]))
v3: Vertex = Vertex(3, np.array([0.0, 1.0, 0.0]))

e0: Edge = Edge(0, np.array([v0, v1]))
e1: Edge = Edge(1, np.array([v1, v2]))
e2: Edge = Edge(2, np.array([v2, v3]))
e3: Edge = Edge(3, np.array([v3, v0]))



e0.physical_tag = 2 # physical tag for BC of pdes on c0
e1.physical_tag = 3 # physical tag for BC of pdes on c0
e2.physical_tag = 4 # physical tag for BC of pdes on c0
e3.physical_tag = 5 # physical tag for BC of pdes on c0

v4: Vertex = Vertex(4, np.array([0.5, 0.2, 0.0]))
v4.physical_tag = 10
v5: Vertex = Vertex(5, np.array([0.5, 0.8, 0.0]))
v5.physical_tag = 10
e4: Edge = Edge(4, np.array([v4, v5]))
e4.physical_tag = 10

# w0: Wire = Wire(4, np.array([e0, e1, e2, e3]), np.array([v0]))
# f0: Face = Face(0, np.array([w0]))
# f0.physical_tag = 1 # physical tag for pdes on c0

domain = create_domain(dimension=max_dim, shapes=np.array([v0, v1, v2, v3, e0, e1, e2, e3, v4, v5, e4]))
domain.build_grahp()
# domain.draw_grahp()

# # compute lines intersections
# # collect lines end points
lines_tool = np.array([e.boundary_points() for e in [e4]])
lines_object = np.array([e.boundary_points() for e in [e0, e1, e2, e3]])
unique_points = lines_lines_intersection(lines_tool, lines_object, deduplicate_points_q=True)

vertices = []
physical_tag_intx = 50
tag = domain.max_tag() + 1
for point in unique_points:
    v: Vertex = Vertex(tag, point)
    v.physical_tag = physical_tag_intx
    vertices.append(v)
    tag += 1
vertices = np.array(vertices)
domain_c1 = create_domain(dimension=max_dim,  shapes=vertices)

# tag = np.max([domain.max_tag(), domain_c1.max_tag()]) + 1
tag = np.max([domain.max_tag()]) + 1
domain_c0 = domain_difference(domain, domain_c1, tag)
# domain_c0.build_grahp()
# domain_c0.draw_grahp()

md_brep_domain = domain_union(domain_c0, domain_c1)
md_brep_domain.build_grahp()
md_brep_domain.draw_grahp()


edges = [shape for shape in md_brep_domain.shapes[1] if shape.physical_tag in [2,3,4,5]]
w0: Wire = Wire(4, edges, np.array([edges[0].boundary_shapes[0]]))
f0: Face = Face(0, np.array([w0]))
f0.physical_tag = 1 # physical tag for pdes on c0

f_edges = [shape for shape in md_brep_domain.shapes[1] if shape.physical_tag in [10]]
f0.immersed_shapes = np.array(f_edges)
md_domain = create_domain(dimension=2,  shapes=np.concatenate(md_brep_domain.shapes))
md_domain.append_shapes(np.array([w0]))
md_domain.append_shapes(np.array([f0]))
all_shapes = 0
all_shapes = 0



# Conformal gmsh discrete representation
h_val = 0.1
domain_h = DiscreteDomain(dimension=md_domain.dimension)
domain_h.domain = md_domain
domain_h.generate_mesh(h_val, 0)
domain_h.write_mesh("gmesh.msh")

# Mesh representation
gmesh = Mesh(dimension=domain.dimension, file_name="gmesh.msh")
gmesh.build_conformal_mesh()
gmesh.write_vtk()

