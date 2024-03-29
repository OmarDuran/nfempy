# Neat finite elements for python

A lightweight library for defining finite elements and simulating multiphysics problems. It is being developed as part of this exploratory project in Python.

The objective of this project is to provide a general, yet simple, method for using finite element methods in Python. 

Some features of the library are
*  Full separation of geometry description, geometry partition, and discrete function spaces;
*  De Rham operators for simplexes and mixed-dimensional PDEs are included in the library; 
*  For conformal-H1, Hdiv, Hcurl, and L2 function spaces, testing includes general projectors of arbitrary high-order in 1D, 2D, and 3D.

Despite the fact that the code structure is highly parallelized, it is still not a trivial task to perform in Python. The entire design can be translated to C++ as a workaround.
