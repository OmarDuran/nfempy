# Python and dependencies
Python version used for the main development is 3.9.12
An exploratory enviroment with python 3.7 is necessary for a clean installation of FEnics and better understanding of the meshes objects

## Publication plan

1. md-coserrat equations, check for convergence and depart from linearized model: Mathematical Models and Methods in Applied Sciences
2. HHO scheme for linear equations POEMS (Conference paper)
3. md-cosserat + fluid
4. md-cosserat + contact mechanics, numerical exploration and stabilization + interior point methods
5. nonlinear md-coserrat


## Dependencies

1. numpy
2. gmsh
4. networkx
5. meshio
6. matplotlib
7. fenics-basix (Fem basis)
8. shapely
9. julia : https://github.com/SciML/LinearSolve.jl (Still not smooth)
10. pyvista: https://docs.pyvista.org/

Shapely will be eliminated.
Firs try for calling LinearSolve.jl inside python <pip install julia> (fail)

# Geometry representation

The geometry cover two main approaches. Fully conformal approximations spaces and broken topolgy for representing plane manifolds or fractures. Simplexes are consider for demonstration. However general polytopal meshes are the goal.

For the case of simplexes:
0d-cells -> points
1d-cells -> lines
2d-cells -> triangles
3d-cells -> tetrahedron

Consider change the mesh data structure to fully support:
Using generic programming for designing a data structure for polyhedral surfaces
Perhaps using https://openmesh-python.readthedocs.io/en/latest/install.html

Perhaps the introduction of an abstract mesh class that represents different mesh data structures is the correct choice

This is more general than the half-edge data structure: "The Dual Half-Edge—A Topological Primal/Dual Data Structure and Construction Operators for Modelling and Manipulating Cell Complexes"

## The half mesh data structure
http://sccg.sk/~samuelcik/dgs/half_edge.pdf


## Guide lines for mesh generation

The mesh construction is given in the following principles. 
A conformal mesh is created using gmsh.

If not fractures are required, the calculatino is straight forward.

Planar fracture are respresented by polygons. All geometrical entities are tag based on precomputed intersections.

Give a set DFN in form of disjoints manifolds, the conformal mesh is adjusted to acomodate (hybrid/mortar variables)

## Topolgy operations on a given gcell object.

Given a geometrical object it must be specified which geometrical entities should be duplicated. Then the connected cells via body connecttions are duplicated and storage into a map (new -> old). 
Now traverse the list of duplicates nodes and updated neighs via conformal mesh graphs.
 

# Quadratures and FEM Basis

Several basis functions are given in [basix](https://github.com/FEniCS/basix). The quadratures are taken from there.

For now FEM hierachical basis are used to have an impression of the  of surface operators. A series of mathematica files were implemented to have preeliminary results.

Approximation spaces of dimension d are created on each cell. The presecen of a operator, pde or variational restriction is associated with physical tags.



# Finite element conformity

For the sake of simplicity only L2-, H1-, Hdiv- and Hcurl-conformal approximation spaces are built on a given material set identified with physical tags.

* Example with H1-conformal scalar laplace.
	* Sprint on, ge, ce

## FEM developments

Current developments steps:

* Homogeneous k-order and cells types
* Only simplexes meshes for now

* Projectors:
	* H1-scalars and -vectors
	* Hdiv-vectors and tensors
* Test cases with
	* R^{n} to R^{3} n in {1,2,3}

Postprocess should be on cells for now.


## DoF mappigns

The construction of a [robuts strategy](https://dl.acm.org/doi/10.1145/3524456) for dof mappings is followed here. However for getting a report closed we only focuse on simplexes of second order.


# Linear solver
In order of importance linear solver are :
 
	1. python available solvers
	2. Julia linear solver : https://github.com/SciML/LinearSolve.jl
	2. pypardiso
	3. petsc4py: consider scipy sparse matrix converted to petsc format

# RoadMap

The main structure of the proyect
### PreProcessor:
	- mesh market -> class MeshMarket

### Geometry:
	- geometry_entity -> class geometry_entity (gen)
	- several geometry descriptions
	- geometry mapping
		- Linear from P1, nonlinear

### Mesh: 
	Mesh -> class Mesh: The mesh needs a user defined physical_tag_map
	Mesh Entity -> mesh_entity (men):
	Conformal Mesher -> class ConformalMesher
	

### Topology:
	- mesh_topology -> class MeshTopology
	- entity_orientation-> class EOrientation
	- entity_permutation-> class EPermutation
	- entity_topology

### FiniteElement:
	- FiniteElement -> class FiniteElement
	- BasisPermutation -> class with only static methods


### FESpaces:
	- DoFMap
	- FESpace: Mesh, MeshTopology, DoFMap
	- FESpaces: Mesh, MeshTopology, list(FESpace), DoFMap
	- Field

### FEForms:
	- L2, H1, Hdiv and Hcurl proyectors
	- Assembler:
	- LinearSolver:
	- L2-error

### PostProcessor:

#### Field:
	From cases:
	* (1) scalar H1/L2-conforming functions
	* (2) vector Hdiv-conforming functions
	* (3) vector Hcurl-conforming functions
	One can notice that
	* (4) From (1) vector, tensor H1/L2-conforming functions
	* (5) From (2) tensor Hdiv-conforming functions
	* (6) From (3) tensor Hcurl-conforming functions
	To unify variables of the kind (1), (2), (3), (4), (5), and (6) the notion of a field should be introduced. 
	A field is a physical quantity, represented by a scalar, vector, or tensor, that has a value for each point in space and time. 
	The abstraction of a field in the context of FE approximations should include:
	* Conformity: [H1, Hcurl, Hdiv] x discontinuous, with discontinuous = [False, True]
	* List of elements: list(FiniteElement)
	* DoFMap
	* The number of subfields: 0 < n_components represents number of repeated instances of cases (1), (2) and (3)
	* k_order: polynomial order of approximation


