

import numpy as np
from topology.domain_market import create_md_box_2D
from mesh.discrete_domain import DiscreteDomain
from mesh.mesh import Mesh

# input
# define a disjoint fracture geometry
points = np.array([
    [0.5, 0.25, 0.0],
    [0.5, 0.75, 0.0],
    [0.25, 0.5, 0.0],
    [0.75, 0.5, 0.0],
])
frac_idx = np.array([
    [0, 1],
    [2, 3],
])
lines = points[frac_idx]

lx = 0.75
ly = 0.75
domain_physical_tags = {"area": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5}
fracture_physical_tags = {"line": 10, "internal_bc": 20, "point": 30}
box_points = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0]])
md_domain = create_md_box_2D(box_points, domain_physical_tags, lines, fracture_physical_tags)


# Conformal gmsh discrete representation
h_val = 0.1
transfinite_agruments = {'n_points': 20, 'meshType': "Bump", 'coef': 0.5}
mesh_arguments = {'lc': h_val, 'n_refinements': 0, 'curves_refinement': ([10], transfinite_agruments)}

domain_h = DiscreteDomain(dimension=md_domain.dimension)
domain_h.domain = md_domain
domain_h.generate_mesh(mesh_arguments)
domain_h.write_mesh("gmesh.msh")

# Mesh representation
gmesh = Mesh(dimension=md_domain.dimension, file_name="gmesh.msh")
gmesh.build_conformal_mesh()
gmesh.write_vtk()

