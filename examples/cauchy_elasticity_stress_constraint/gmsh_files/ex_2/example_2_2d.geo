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
Line{300, 400} In Surface {all_surfaces[]};

// STEP 3: Finalize the geometry to ensure conformality.
Coherence;

//------------------------------------------------------------------------------------
// 4. Robust Physical Group Assignment
//------------------------------------------------------------------------------------

// --- Define seed points for each of the 6 faulted blocks ---
p_bl = {100, 100, 0}; p_br = {800, 100, 0};
p_ml = {100, 225, 0}; p_mr = {800, 225, 0};
p_tl = {100, 400, 0}; p_tr = {800, 400, 0};

// --- Find surfaces at seed points and assign to layer groups ---
s_bl = Surface At Point p_bl; Physical Surface("underburden_layer", 1) = {s_bl};
s_br = Surface At Point p_br; Physical Surface("underburden_layer", 1) += {s_br};
s_ml = Surface At Point p_ml; Physical Surface("reservoir_layer", 2) = {s_ml};
s_mr = Surface At Point p_mr; Physical Surface("reservoir_layer", 2) += {s_mr};
s_tl = Surface At Point p_tl; Physical Surface("overburden_layer", 3) = {s_tl};
s_tr = Surface At Point p_tr; Physical Surface("overburden_layer", 3) += {s_tr};

// --- Identify and group the line fragments ---
// We select all fragments of a feature by searching for lines within its original bounding box.
fault_lines[] = Line In BoundingBox{fault_x_position-1, -1, -1, fault_x_position+300, box_height+1, 1};
Physical Line("fault", 200) = {fault_lines[]};

well1_lines[] = Line In BoundingBox{well1_end_x-1, well1_end_y-1, -1, well1_start_x+1, box_height+1, 1};
Physical Line("wellbore_1", 300) = {well1_lines[]};

well2_lines[] = Line In BoundingBox{well2_start_x-1, well2_end_y-1, -1, well2_kink_x+51, box_height+1, 1};
Physical Line("wellbore_2", 400) = {well2_lines[]};

// --- Identify and group the outer boundaries ---
b_bottom[] = Line In BoundingBox {-1, -1, -1, box_width+1, 1, 1};
b_top[] = Line In BoundingBox {-1, box_height-1, -1, box_width+1, box_height+1, 1};
b_left[] = Line In BoundingBox {-1, -1, -1, 1, box_height+1, 1};
b_right[] = Line In BoundingBox {box_width-1, -1, -1, box_width+1, box_height+1, 1};
Physical Line("bottom_boundary", 201) = {b_bottom[]};
Physical Line("top_boundary", 202) = {b_top[]};
Physical Line("left_boundary", 203) = {b_left[]};
Physical Line("right_boundary", 204) = {b_right[]};