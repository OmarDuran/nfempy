# Boolean operations for shapes:
Three main boolean operations are supported:

- Intersection or common parts
- Difference or Cut or -
- Union or Fusion or +

# Notes on domains:
The main idea is to have graph representation of the domains in order to:

- Convert a graph to a domain and vice versa
- Convert a graph to a mesh and vice versa
In both cases above, the convertion operates in boundary representation (BRep) domains and geometries. No boundary representation of shapes (Non-BRep) like embedded or immersed shapes are removed and aded, respectively, before and after the operation. 

For md-domains and meshes, the natural steps are:

- 1D: ongoing
- 2D: 
- 3D: 

## 0D embedding in 1D
The functionality is fully supported by the shapes. However, gmsh does not peform the embedding with `gmsh.model.mesh.embed(0, tags_0d, 1, self.stride_tag(1, curve.tag))`. Therefore any point intended to be embedded will be added as a tool while computing the difference and as consequences 1D domains will not include embedded points.

## meshing a mixed-dimensional domain `md-Omega`:
The proceedure follows the main general steps:

- Initialize a structure for shape collection: `shapes = [{}, {}]`
- Define domain definition (`Omega`) as fixed-dimenional domain
- Define subdomains to substract (`Omega^{c}_{idx}`)
- Compute `Omega - Omega^{c}_{idx}` and collect resulting parts
- Build a Domain for `Omega - Omega^{c}_{idx}` as broken domain
- Perform the union of of `md-Omega = Omega - Omega^{c}_{idx}` + `Omega^{c}_{idx}`
- Build md-Omega_h via `md_Omega_h = DiscreteDomain(dimension=domain.dimension)`
- Write a vtk representation of `md_Omega_h`



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