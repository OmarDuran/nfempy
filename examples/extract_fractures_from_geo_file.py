import numpy as np
import gmsh

gmsh.initialize()

gmsh.open("fracture_files/network_3.geo")

print("Model " + gmsh.model.getCurrent() + " (" + str(gmsh.model.getDimension()) + "D)")
entities_c1 = gmsh.model.getEntities(dim=2)
entities_c3 = gmsh.model.getEntities(dim=0)

fractures = {}
fracture_id = 0
for e in entities_c1:
    # Dimension and tag of the entity:
    dim = e[0]
    tag = e[1]

    boundary = gmsh.model.get_boundary(
        [e], combined=True, oriented=True, recursive=True
    )
    if len(boundary) != 4:
        continue
    points = np.vstack(
        [gmsh.model.occ.get_bounding_box(0, pair[1])[0:3] for pair in boundary]
    )
    # # get curves loops
    # loops = gmsh.model.occ.get_curve_loops(tag)
    #
    # node_ids = []
    # for loop in loops:
    #     # get nodes
    #     gmsh.model.get_boundary()
    #     if not isinstance(loop[0], np.ndarray) :
    #         continue
    #     if len(loop[0]) != 4:
    #         continue
    #     for tag in loop[0]:
    #         nodes = np.hstack(gmsh.model.get_adjacencies(0, tag))
    #         node_ids.append(nodes)
    #
    # if len(node_ids) == 0:
    #     continue
    #
    # node_ids = np.unique([node for pair in node_ids for node in pair])
    print("fracture_id: ", fracture_id)
    fractures.__setitem__(fracture_id, points)
    fracture_id += 1

data = np.vstack(tuple([fractures[key].flatten() for key in fractures.keys()]))

aka = 0

# We can use this to clear all the model data:
gmsh.clear()

gmsh.finalize()
