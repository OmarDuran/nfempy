# Boolean operations for shapes:
Three main boolean operations are supported:

- Intersection or common parts
- Difference or Cut or -
- Union or Fusion or +

# Notes on domains:
The general idea is to have graph representation of the domains in order to

- Convert a graph to a domain and vice versa
- Convert a graph to a mesh and vice versa

For md-domains and meshes, the natural steps are:

- 1D: ongoing
- 2D: 
- 3D: 

# Notes on intersections:
This operation may generate new shapes and rely in geometrical information. Based on the information given in [[Open cascade]](https://dev.opencascade.org/doc/overview/html/specification__boolean_operations.html). No pave blocks / sectors are supported.

# Difference or Cut or -:
This operation is restricted for shapes having a boundary representation  only to boundary representation (BRep). In other words, only operations between shapes with co_dimension difference of one are permitted. The following functionalities may be desired:

- New parts will are going to inherit the physical tag from the object shape



## Check list for intersections
Marked items should be understood as tested.
## Vertex - shapes intersection

- [x] Vertex - Vertex intersection
- [x] Vertex - Edge intersection
- [ ] Vertex - Wire intersection
- [ ] Vertex - Face intersection
- [ ] Vertex - Shell intersection
- [ ] Vertex - Solid intersection

## Edge - shapes intersection

- [x] Edge - Vertex intersection
- [x] Edge - Edge intersection
- [ ] Edge - Wire intersection
- [ ] Edge - Face intersection
- [ ] Edge - Shell intersection
- [ ] Edge - Solid intersection