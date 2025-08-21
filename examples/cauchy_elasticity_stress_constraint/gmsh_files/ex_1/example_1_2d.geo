// Gmsh .geo script for a 2D faulted anticline structure.
// VERSION 3: Corrected geometry definition for robustness and a thinner middle layer.
//
// This model features:
// 1. A rectangular domain with three geological layers.
// 2. The internal layers are folded into an anticline shape.
// 3. A normal fault dipping at 60 degrees cuts through the entire model.

SetFactory("OpenCASCADE");

//------------------------------------------------------------------------------------
// 1. Parameters
//------------------------------------------------------------------------------------
number_of_points = 25;
number_of_points_right = 25;
number_of_points_left = 15;
number_of_points_res_fault = 8;

// --- Domain dimensions
box_width = 1000.0;
box_height = 500.0;

// --- Anticline parameters for a THINNER MIDDLE LAYER ---
// The middle layer will now have a thickness of 50m.
interface1_y_side = 150.0; // Bottom of middle layer at sides
anticline_crest1_y = 250.0; // Bottom of middle layer at crest
interface2_y_side = 200.0; // Top of middle layer at sides (150 + 50)
anticline_crest2_y = 300.0; // Top of middle layer at crest (250 + 50)

// --- Fault parameters
// Defines a fault dipping 60 degrees from horizontal.
fault_start_x = 100.0;
fault_start_y = box_height;
fault_end_x = fault_start_x + (box_height / Tan(60*Pi/180.0));
fault_end_y = 0.0;

// --- Meshing
lcar = 20.0; // General mesh size
lcar_fault = 10.0; // Finer mesh size along the fault

//------------------------------------------------------------------------------------
// 2. Geometry Definition (Layers and Fault)
//------------------------------------------------------------------------------------

// --- Points ---
// Box corners
Point(1) = {0, 0, 0, lcar};
Point(2) = {box_width, 0, 0, lcar};
Point(3) = {box_width, box_height, 0, lcar};
Point(4) = {0, box_height, 0, lcar};
// Interface 1 (bottom) points
Point(5) = {0, interface1_y_side, 0, lcar};
Point(6) = {box_width, interface1_y_side, 0, lcar};
Point(7) = {box_width/2, anticline_crest1_y, 0, lcar};
// Interface 2 (top) points
Point(8) = {0, interface2_y_side, 0, lcar};
Point(9) = {box_width, interface2_y_side, 0, lcar};
Point(10) = {box_width/2, anticline_crest2_y, 0, lcar};

// --- Lines ---
// Create the curved interfaces using splines
Spline(1) = {5, 7, 6}; // Bottom anticline interface
Spline(2) = {8, 10, 9}; // Top anticline interface
// Straight lines for the box, segmented by the interfaces
Line(3) = {1, 2}; // Bottom boundary
Line(4) = {3, 4}; // Top boundary
Line(5) = {1, 5}; // Left side, bottom segment
Line(6) = {5, 8}; // Left side, middle segment
Line(7) = {8, 4}; // Left side, top segment
Line(8) = {2, 6}; // Right side, bottom segment
Line(9) = {6, 9}; // Right side, middle segment
Line(10) = {9, 3}; // Right side, top segment

// --- Surfaces for the unfaulted Geological Layers ---
// Create each layer with its own clean, closed loop.
Line Loop(1) = {3, 8, -1, -5};   // Bottom layer
Line Loop(2) = {1, 9, -2, -6};   // Middle layer
Line Loop(3) = {2, 10, 4, -7};  // Top layer
Plane Surface(101) = {1}; // Bottom layer (use high tags for clarity)
Plane Surface(102) = {2}; // Middle layer
Plane Surface(103) = {3}; // Top layer

// --- Fault Definition ---
Point(11) = {fault_start_x, fault_start_y, 0, lcar_fault};
Point(12) = {fault_end_x, fault_end_y, 0, lcar_fault};
Line(200) = {11, 12}; // The fault line

//------------------------------------------------------------------------------------
// 3. Intersect Layers with the Fault (Boolean Operation)
//------------------------------------------------------------------------------------
// We fragment the three surfaces (101, 102, 103) with the fault line (200).
// The `Delete` keyword removes the original entities, leaving only the fragments.
BooleanFragments { Surface{101, 102, 103}; Delete; }{ Line{200}; Delete; }


Transfinite Line { 5,11 } = number_of_points_right;
Transfinite Line { 2,9 } = number_of_points_left;
Transfinite Line { 15,3 } = number_of_points;
Transfinite Line { 10 } = number_of_points_res_fault;



//------------------------------------------------------------------------------------
// 4. Physical Groups
//------------------------------------------------------------------------------------
// After fragmentation, we identify the new pieces to assign them to physical groups.
// The `Surface In BoundingBox` command is a convenient way to select the new surfaces.

// Identify fragments for the bottom layer
s_bottom[] = {1,2};
Physical Surface("underburden_layer", 1) = {s_bottom[]};

// Identify fragments for the middle layer
s_middle[] = {3,4};
Physical Surface("reservoir_layer", 2) = {s_middle[]};

// Identify fragments for the top layer
s_top[] = {5, 6};
Physical Surface("overburden_layer", 3) = {s_top[]};


// Identify the outer boundary fragments
b_bottom[] = Line In BoundingBox {-0.1, -0.1, -0.1, box_width+0.1, 0.1, 0.1};
b_top[] = Line In BoundingBox {-0.1, box_height-0.1, -0.1, box_width+0.1, box_height+0.1, 0.1};
b_left[] = Line In BoundingBox {-0.1, -0.1, -0.1, 0.1, box_height+0.1, 0.1};
b_right[] = Line In BoundingBox {box_width-0.1, -0.1, -0.1, box_width+0.1, box_height+0.1, 0.1};
Physical Line("bottom_boundary", 4) = {b_bottom[]};
Physical Line("top_boundary", 5) = {b_top[]};
Physical Line("left_boundary", 6) = {b_left[]};
Physical Line("right_boundary", 7) = {b_right[]};

// Identify the fault line fragments
fault_lines[] = {3,10,15};
Physical Line("fault", 8) = {fault_lines[]};

//------------------------------------------------------------------------------------
// 5. Meshing
//------------------------------------------------------------------------------------
// Since the geometry is now complex and unstructured due to the fault,
// we do not use the Transfinite algorithm.