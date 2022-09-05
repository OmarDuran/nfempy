# Python and dependencies
Python version used for the main development is 3.9.12
An exploratory enviroment with python 3.7 is necessary for a clean installation of FEnics and better understanding of the meshes objects

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

MacDown is open source and is a volunteer effort. This means that it depends on people to give some of their free time to improve it and make it even better.

If you are reading this, then you are probably curious or want to contribute in some way. Read on to see how you can do so.

## Guide lines for mesh generation

The mesh construction is given in the following principles. 
A conformal mesh is created using gmsh.

If not fractures are required a boundary mesh is created to incorporated static BC.

Planar fracture are respresented by polygons. All geometrical entities are tag based on precomputed intersections.

# Quadratures and FEM Basis

Several basis functions are given in [basix](https://github.com/FEniCS/basix). The quadratures are taken from there.

For now FEM hierachical basis are used to have an impression of the  of surface operators. A series of mathematica files were implemented to have preeliminary results.


# DoF mappigns

The construction of a [robuts strategy](https://dl.acm.org/doi/10.1145/3524456) for dof mappings is followed here

# Linear solver
 linear solver are :
 
	1. python available solvers
	2. pypardiso
	3. petsc4py

# Benchmarks	

