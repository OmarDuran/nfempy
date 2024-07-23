import pickle
import numpy as np
from mesh.mesh import Mesh
from topology.domain_market import build_box_3D
from mesh.discrete_domain import DiscreteDomain
from geometry.operations.polygon_geometry_operations import polygons_polygons_intersection

from geometry.operations.point_geometry_operations import points_line_intersection
from geometry.operations.line_geometry_operations import lines_lines_intersection
from mesh.mesh_operations import cut_conformity_along_c1_lines


# input
# define a disjoint fracture geometry
points = np.array(
    [
        [-0.809017, -0.587785, 0.5],
        [-0.809017, 0.587785, 0.5],
        [0.309017, 0.951057, 0.5],
        [1.0, 0.0, 0.5],
        [0.309017, -0.951057, 0.5],
        [-0.76128112,  0.35810633,  0.78290468],
        [-0.25224438,  1.09283758,  0.01934957],
        [ 0.74358251,  0.5936982 , -0.35635677],
        [ 0.85      , -0.44951905,  0.175     ],
        [-0.08005701, -0.59512305,  0.87910252],
        [0.5, -0.387785, -0.509017],
        [0.5, 0.787785, -0.509017],
        [0.5, 1.151057, 0.609017],
        [0.5, 0.2, 1.3],
        [0.5, -0.751057, 0.609017],
    ]
)
frac_idx = np.array(
    [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
)
polygons = points[frac_idx]

lx = 1.0
ly = 1.0
lz = 1.0
domain_physical_tags = {"solid": 1, "bc_0": 2, "bc_1": 3, "bc_2": 4, "bc_3": 5, "bc_4": 6, "bc_5": 7}
fracture_physical_tags = {"line": 10, "internal_bc": 20, "point": 30}
box_points = np.array([
    [0, 0, 0],
    [lx, 0, 0],
    [lx, ly, 0],
    [0, ly, 0],
    [0, 0, lz],
    [lx, 0, lz],
    [lx, ly, lz],
    [0, ly, lz]
]
)

fracture_data = {
    "geometry": polygons,
    "fracture_physical_tags": fracture_physical_tags,
}
# Specify the filename
filename = "fracture_data.pkl"
# Write the object to a file using pickle
with open(filename, "wb") as file:
    pickle.dump(fracture_data, file)

md_domain = build_box_3D(
    box_points, domain_physical_tags
)

# computing boundary_lines
boundary_polygons = []
for shape in md_domain.shapes[2]:
    if shape.composite:
        continue
    boundary_polygons.append(shape.boundary_points())
boundary_polygons = np.array(boundary_polygons)

# compute polygon intersections
boundary_intx_lines = polygons_polygons_intersection(polygons, boundary_polygons,deduplicate_lines_q=False)
fracture_intx_lines = polygons_polygons_intersection(polygons, polygons,deduplicate_lines_q=False)

aka = 0
# intx_lines = np.array(intx_data)
#
# # compute line intersections
# fracture_intx = lines_lines_intersection(lines_tools=intx_lines, lines_objects=intx_lines, deduplicate_points_q=True)
#
# # make line mesh conformal
# boundary_vertices = {}
# for i, line in enumerate(intx_lines):
#     a, b = line
#     out, intx_idx = points_line_intersection(fracture_intx, a, b)
#     if len(out) == 0:
#         continue
#     boundary_vertices[i] = out
#
# # create facets
# s0 = polygons[0]
#
#
# # Conformal gmsh discrete representation
# h_val = 1.0
# transfinite_agruments = {"n_points": 2, "meshType": "Bump", "coef": 1.0}
# mesh_arguments = {
#     "lc": h_val,
#     "n_refinements": 0,
#     "curves_refinement": ([10], transfinite_agruments),
# }
#
# domain_h = DiscreteDomain(dimension=md_domain.dimension)
# domain_h.domain = md_domain
# domain_h.generate_mesh(mesh_arguments)
# domain_h.write_mesh("gmesh.msh")
#
# # Mesh representation
# gmesh = Mesh(dimension=md_domain.dimension, file_name="gmesh.msh")
# gmesh.build_conformal_mesh()
# gmesh.write_vtk()


# physical_tags = {"c1": 10, "c1_clones": 50}
# lines = fracture_data["geometry"]
# physical_tags = fracture_data["fracture_physical_tags"]
# physical_tags["line_clones"] = 50
# physical_tags["point_clones"] = 100
# # cut_conformity_along_c1_lines(lines, physical_tags, gmesh)
# gmesh.write_vtk()
