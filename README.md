# N-dimensional finite elements for python

A lightweight library for defining mixed dimensional (N-dimension) finite elements and simulating multiphysics problems. It is being developed as a prototype tool in Python.

The objective of this project is to provide a general, yet simple, method for using finite element methods in Python. 

Some features of the library are
*  Full separation of geometry description, geometry partition, and discrete function spaces;
*  De Rham operators for simplexes and mixed-dimensional PDEs are included in the library; 
*  For conformal-H1, Hdiv, Hcurl, and L2 function spaces, testing includes general projectors of arbitrary high-order in 1D, 2D, and 3D.

Although the code is highly parallelized, in Python this is not an easy task to accomplish cost-effectively. 
