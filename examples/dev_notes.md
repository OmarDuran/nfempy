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
9. dill: because serialization and parallelization
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
Non-conformal mesh is not part of the package priorities.
 

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

### Projectors available:
Projectors:
The proyectors are essential in verfiying the correctness of functions with desired conformity 
for d in {0,1,2,3}:
	L2-scalars: d in {0,1,2,3}
	L2-vectors: d in {0,1,2,3}
	L2-tensors: d in {0,1,2,3}
	H1-scalars: d in {0,1,2,3}
	H1-vectors: d in {0,1,2,3}
	H1-tensors: d in {0,1,2,3}
	Hdiv-vectors: d in {1,2,3}
	Hcurl-vectors: d in {1,2,3}
	Hdiv-tensors
	Hcurl-tensors

### PostProcessor:

#### Discrete Field:
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

### Notes on geometry representation and processing:
The following refactors are needed: https://dev.opencascade.org/sites/default/files/pdf/Topology.pdf
Geometrical objects represented by GeometryCell (Shapes) should consider immersed_entities and boundary entities  	
Shapes:
	Vertex: actual point in R3
	Edge: part of a curve limited by vertices (This could contain intersections).
	Wire: Set of Edges connected by edges (topological information)
	Face: part of a surface limited by wires (This could contain intersections).
	Shell: Set of faces connected by edges
	Solid: a part of the space (subdomain) limited by shells  (This could contain intersections).
	CompositeSolid: set of solids connected by their faces
	
Domain: set of any topological shape
	
The skins will be extra information after a Composite is created
	
#### TODO: Rename GeometryCell to GeometryEntity

### Notes on assembler implementation:
The best local vectorization occurs in eliminating integration point loops

### Log second-order excecution time:
	
	Note: In a 3d unit cube, the mesh generation for h = 1/64 will generate 200873 nodes and 1227986 elements (tetrahedra) and memory usage will be around 8Gb. So it is not worthy to optimize beyond  h = 1/32 because the mesh generation will be untractable. Ports for other mesh generators is required. 
	The case h = 1/32 gmsh will generate 27364 nodes and 162993 elements (tetrahedra) will be the limitation of the approach with python.
	
	
	Performance for
	- 3D laplacian operator
	- six component field
	- h = 1/16
	- k_order = 3
	
	h-size:  0.0625
	Field:: DoFMap construction time: 1.7416038513183594 seconds
	Field:: Number of processed elements: 18842
	Field:: Element construction time: 128.50061893463135 seconds
	Field:: DoFMap construction time: 0.31716299057006836 seconds
	Field:: Number of processed elements: 3700
	Field:: Element construction time: 7.492000102996826 seconds
	n_dof:  557622
	Triplets creation time: 0.08223390579223633 seconds
	Assembly time: 135.04999828338623 seconds
	Linear solver time: 598.3841590881348 seconds
	L2-error time: 9.504310131072998 seconds
	L2-error:  6.91696585636745e-08
	Post-processing time: 7.020235061645508 seconds
	
	Performance for:
	- 3D second-order Cosserat operator
	- Six component field
	- h = 1/1
	- l = 3
	- k_order = 2
	- 	
	h-size:  1.0
	l-refi:  3
	DiscreteField:: DoFMap construction time: 0.8933699131011963 seconds
	DiscreteField:: Number of processed elements: 12288
	DiscreteField:: Elements construction time: 46.39259123802185 seconds
	DiscreteField:: Number of processed bc elements: 1536
	DiscreteField:: Boundary Elements construction time: 2.3282089233398438 seconds
	n_dof:  107814
	Triplets creation time: 0.09861302375793457 seconds
	Assembly time: 174.8708040714264 seconds
	Linear solver time: 30.13961410522461 seconds
	L2-error time: 2.045224905014038 seconds
	L2-error:  5.954808931501035e-05
	Post-processing time: 2.6556379795074463 seconds
	
	Performance for (Simulation not complete):
	- 3D second-order Cosserat operator
	- Six component field
	- h = 1/1
	- l = 4
	- k_order = 2
	
	h-size:  1.0
	l-refi:  4
	DiscreteField:: DoFMap construction time: 9.730278968811035 seconds
	DiscreteField:: Number of processed elements: 98304
	DiscreteField:: Elements construction time: 334.8427629470825 seconds
	DiscreteField:: Number of processed bc elements: 6144
	DiscreteField:: Boundary Elements construction time: 8.834323406219482 seconds
	n_dof:  823878
	Triplets creation time: 4.465027093887329 seconds
	Assembly time: 1393.5477249622345 seconds


