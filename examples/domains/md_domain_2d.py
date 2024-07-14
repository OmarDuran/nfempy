

import numpy as np
from mesh.mesh import Mesh

# Mesh representation
gmesh = Mesh(dimension=2, file_name="gmesh.msh")
gmesh.build_conformal_mesh()
gmesh.write_vtk()

physical_tags = {'c1': 10, 'c1_clones': 50}
gmesh.cut_c1_conformity_physical_tags(physical_tags)
gmesh.write_vtk()
# gmesh.draw_graph(gc1)
aka = 0



