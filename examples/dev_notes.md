# Python and dependencies
Python version used for the main development is 3.9.12
An exploratory enviroment with python 3.7 is necessary for a clean installation of FEnics and better understanding of the meshes objects

## Publication plan

1) md-coserrat equations, check for convergence and depart from linearized model: Mathematical Models and Methods in Applied Sciences
2) HHO scheme for linear equations POEMS (Conference paper)
3) md-coserrat + contact mechanics, numerical exploration and stabilization + interior point methods
4) nonlinear md-coserrat + fluid


## Dependencies

1. numpy
2. gmsh
4. networkx
5. meshio
6. matplotlib
7. fenics-basix
8. shapely

Shapely will be eliminated.

# Geometry representation

The geometry cover two main approaches. Fully conformal approximations spaces and broken topolgy for representing plane manifolds or fractures. Simplexes are consider for demonstration. However general polytopal meshes are the goal.

For the case of simplexes:
0d-cells -> points
1d-cells -> lines
2d-cells -> triangles
3d-cells -> tetrahedron


## Guide lines for mesh generation

The mesh construction is given in the following principles. 
A conformal mesh is created using gmsh.

If not fractures are required a boundary mesh is created to incorporate BC.

Planar fracture are respresented by polygons. All geometrical entities are tag based on precomputed intersections.

## Topolgy operations on a given gcell object.

Given a geometrical object it must be specified which geometrical entities should be duplicated. Then the connected cells via body connecttions are duplicated and storage into a map (new -> old). 
Now traverse the list of duplicates nodes and updated neighs via conformal mesh graphs.
 
For the case of fractures.

# Quadratures and FEM Basis

Several basis functions are given in [basix](https://github.com/FEniCS/basix). The quadratures are taken from there.

For now FEM hierachical basis are used to have an impression of the  of surface operators. A series of mathematica files were implemented to have preeliminary results.

Approximation spaces of dimension d are created on each cell. The presecen of a operator, pde or variational restriction is associated with physical tags.



# Finite element conformity

For the sake of simplicity only L2-, H1- and Hdiv-conformal approximation spaces are built on a given material set identified with physical tags.

* Example with H1-conformal scalar laplace

## FEM developments

Current developments steps:

* Projectors:
	* H1-scalars and -vectors
	* Hdiv-vectors and tensors
* Test cases with
	* R^{n} to R^{3} n in {1,2,3}

Postprocess should be on cells for now.


## DoF mappigns

The construction of a [robuts strategy](https://dl.acm.org/doi/10.1145/3524456) for dof mappings is followed here. However for getting a report closed we only focuse on simplexes of second order.


# Linear solver
 linear solver are :
 
	1. python available solvers
	2. pypardiso
	3. petsc4py: consider scipy sparse matrix converted to petsc format

# Benchmarks	

