

import numpy as np
import pickle
from mesh.mesh import Mesh
from mesh.mesh_operations import cut_conformity_along_c1_lines

# Specify the filename
filename = 'fracture_data.pkl'
# Read the object from the file using pickle
with open(filename, 'rb') as file:
    fracture_data = pickle.load(file)

# Mesh representation
gmesh = Mesh(dimension=2, file_name="gmesh.msh")
gmesh.build_conformal_mesh()
gmesh.write_vtk()

physical_tags = {'c1': 10, 'c1_clones': 50}
lines = fracture_data['geometry']
physical_tags = fracture_data['fracture_physical_tags']
physical_tags['line_clones'] = 50
physical_tags['point_clones'] = 100
cut_conformity_along_c1_lines(lines,physical_tags,gmesh)
# gmesh.cut_c1_conformity_physical_tags(physical_tags)
gmesh.write_vtk()
# gmesh.draw_graph(gc1)
aka = 0



