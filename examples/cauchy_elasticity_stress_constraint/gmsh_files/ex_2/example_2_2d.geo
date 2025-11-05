// Gmsh .geo script for a 2D faulted anticline with two deviated wellbores.
// FINAL ROBUST VERSION: Uses the correct paradigm of cutting with the fault and
// embedding the wells for a guaranteed conformal mesh.

SetFactory("OpenCASCADE"); // Use the advanced geometry kernel

//------------------------------------------------------------------------------------
// 1. Parameters
//------------------------------------------------------------------------------------
box_width = 1000.0;
box_height = 500.0;

// --- Anticline parameters
interface1_y_side = 150.0;
anticline_crest1_y = 250.0;
interface2_y_side = 200.0; // 50m thickness at side
anticline_crest2_y = 300.0; // 50m thickness at crest

// --- Fault parameters (60-degree dip)
fault_x_position = 300.0;

// --- Wellbore 1 Trajectory parameters ---
well1_start_x = 800.0; // X-position where well starts at the top
well1_kink_x = 750.0;  // Intermediate point to control curvature
well1_kink_y = 350.0;
well1_end_x = 550.0;   // Well terminates in the middle layer
well1_end_y = 255.0;

// --- NEW: Wellbore 2 Trajectory parameters ---
well2_start_x = 100.0; // Starts on the left side
well2_kink_x = 150.0;
well2_kink_y = 350.0;
well2_end_x = 300.0;   // Terminates in the bottom layer
well2_end_y = 250.0;

// --- Meshing parameters
lcar = 50.0; // General mesh size
lcar_fault = 10.0; // Finer mesh size on the fault
lcar_well = 10.0;   // Finer mesh size along the wellbores

//------------------------------------------------------------------------------------
// 2. Initial Geometry Definition
//------------------------------------------------------------------------------------

// --- Define Points and Lines for Layers and Fault ---
Point(1) = {0, 0, 0, lcar}; Point(2) = {box_width, 0, 0, lcar};
Point(3) = {box_width, box_height, 0, lcar}; Point(4) = {0, box_height, 0, lcar};
Point(5) = {0, interface1_y_side, 0, lcar}; Point(6) = {box_width, interface1_y_side, 0, lcar};
Point(7) = {box_width/2, anticline_crest1_y, 0, lcar};
Point(8) = {0, interface2_y_side, 0, lcar}; Point(9) = {box_width, interface2_y_side, 0, lcar};
Point(10) = {box_width/2, anticline_crest2_y, 0, lcar};
Spline(1) = {5, 7, 6}; Spline(2) = {8, 10, 9};
Line(3) = {1, 2}; Line(4) = {4, 3}; Line(5) = {1, 5}; Line(6) = {5, 8};
Line(7) = {8, 4}; Line(8) = {2, 6}; Line(9) = {6, 9}; Line(10) = {9, 3};
Line Loop(1) = {3, 8, -1, -5}; Plane Surface(101) = {1};
Line Loop(2) = {1, 9, -2, -6}; Plane Surface(102) = {2};
Line Loop(3) = {2, 10, -4, -7}; Plane Surface(103) = {3};
Point(11) = {fault_x_position, box_height, 0, lcar_fault};
Point(12) = {fault_x_position + box_height/Tan(60*Pi/180), 0, 0, lcar_fault};
Line(200) = {11, 12}; // Fault line

// --- Define the Wellbore Trajectories ---
// Wellbore 1
Point(20) = {well1_start_x, box_height, 0, lcar_well};
Point(21) = {well1_kink_x, well1_kink_y, 0, lcar_well};
Point(22) = {well1_end_x, well1_end_y, 0, lcar_well};
Spline(300) = {20, 21, 22};

// Wellbore 2
Point(30) = {well2_start_x, box_height, 0, lcar_well};
Point(31) = {well2_kink_x, well2_kink_y, 0, lcar_well};
Point(32) = {well2_end_x, well2_end_y, 0, lcar_well};
Spline(400) = {30, 31, 32};

//------------------------------------------------------------------------------------
// 3. Cut Geometry with Fault, then Embed Wells
//------------------------------------------------------------------------------------

// STEP 1: Cut the layers with the fault line.
BooleanFragments { Surface{101, 102, 103}; Delete; }{ Line{200, 300, 400}; Delete; }

// STEP 2: Embed the wellbores into the newly faulted surfaces.
all_surfaces[] = Surface{:};
//Line{300, 400} In Surface {all_surfaces[]};

// STEP 3: Finalize the geometry to ensure conformality.
Coherence;

npts_bottom = 40;
npts_top = 10;
npts_fault = 10;
npts_fault_res = 5;
npts_well_left = 10;
npts_well_right = 10;

Transfinite Line { 2, 5 } = npts_bottom;
Transfinite Line { 9, 10, 13, 14} = npts_top;
Transfinite Line { 3, 21 } = npts_fault;
Transfinite Line { 11 } = npts_fault_res;
Transfinite Line { 19 } = npts_well_left;
Transfinite Line { 23 } = npts_well_right;


// --- Find surfaces at seed points and assign to layer groups ---
Physical Surface("underburden_layer", 1) = {1, 2};
Physical Surface("reservoir_layer", 2) = {3, 4};
Physical Surface("overburden_layer", 3) = {5, 6, 7, 8};

Physical Line("bottom_boundary", 4) = {4, 7};
Physical Line("top_boundary", 5) = {18, 20, 22, 24};
Physical Line("left_boundary", 6) = {1, 8, 17};
Physical Line("right_boundary", 7) = {6, 15, 25};

Physical Line("well_1", 8) = {16, 23};
Physical Line("well_2", 9) = {12, 19};
Physical Line("fault", 10) = {21, 11, 3};
