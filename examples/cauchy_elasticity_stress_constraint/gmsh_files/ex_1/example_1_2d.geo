// Gmsh .geo script for a simple 2D anticline structure.
// This is a simplified base model with three curved geological layers
// inside a rectangular domain.

//------------------------------------------------------------------------------------
// 1. Parameters
//------------------------------------------------------------------------------------
box_width = 1000.0;
box_height = 500.0;

// --- Anticline parameters
interface1_y_side = 150.0; // Bottom of middle layer at sides
anticline_crest1_y = 250.0; // Bottom of middle layer at crest
interface2_y_side = 200.0; // Top of middle layer at sides (50m thick)
anticline_crest2_y = 300.0; // Top of middle layer at crest (50m thick)

// --- Meshing parameters
lcar = 50.0; // General mesh size

//------------------------------------------------------------------------------------
// 2. Geometry Definition
//------------------------------------------------------------------------------------

// --- Define Points for boundaries and anticline interfaces ---
Point(1) = {0, 0, 0, lcar};
Point(2) = {box_width, 0, 0, lcar};
Point(3) = {box_width, box_height, 0, lcar};
Point(4) = {0, box_height, 0, lcar};
Point(5) = {0, interface1_y_side, 0, lcar};
Point(6) = {box_width, interface1_y_side, 0, lcar};
Point(7) = {box_width/2, anticline_crest1_y, 0, lcar};
Point(8) = {0, interface2_y_side, 0, lcar};
Point(9) = {box_width, interface2_y_side, 0, lcar};
Point(10) = {box_width/2, anticline_crest2_y, 0, lcar};

// --- Define Lines for boundaries and interfaces ---
Spline(1) = {5, 7, 6}; // Bottom anticline interface
Spline(2) = {8, 10, 9}; // Top anticline interface
Line(3) = {1, 2};     // Bottom boundary
Line(4) = {4, 3};     // Top boundary
Line(5) = {1, 5};     // Left side, bottom segment
Line(6) = {5, 8};     // Left side, middle segment
Line(7) = {8, 4};     // Left side, top segment
Line(8) = {2, 6};     // Right side, bottom segment
Line(9) = {6, 9};     // Right side, middle segment
Line(10) = {9, 3};    // Right side, top segment

// --- Define the three layer surfaces ---
// Each layer is defined by its own closed loop of lines.
Line Loop(1) = {3, 8, -1, -5};   // Bottom layer
Line Loop(2) = {1, 9, -2, -6};   // Middle layer
Line Loop(3) = {2, 10, -4, -7};  // Top layer
Plane Surface(101) = {1}; // Bottom layer
Plane Surface(102) = {2}; // Middle layer
Plane Surface(103) = {3}; // Top layer

//------------------------------------------------------------------------------------
// 3. Physical Group Assignment
//------------------------------------------------------------------------------------
// Assign names and unique integer tags to the layers and boundaries.
// This is crucial for applying material properties and boundary conditions in simulations.

Physical Surface("underburden_layer", 1) = {101};
Physical Surface("reservoir_layer", 2) = {102};
Physical Surface("overburden_layer", 3) = {103};

Physical Line("bottom_boundary", 4) = {3};
Physical Line("top_boundary", 5) = {4};
Physical Line("left_boundary", 6) = {5, 6, 7};
Physical Line("right_boundary", 7) = {8, 9, 10};
