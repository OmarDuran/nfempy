// Gmsh project for a unit square domain split into four square sections
SetFactory("OpenCASCADE");

// Define a characteristic length for the mesh size
lc = 0.125;

// --- 1. GEOMETRY DEFINITION ---

// Define corner points of the unit square
Point(1) = {0, 0, 0, lc}; // Bottom-left
Point(2) = {1, 0, 0, lc}; // Bottom-right
Point(3) = {1, 1, 0, lc}; // Top-right
Point(4) = {0, 1, 0, lc}; // Top-left

// Define the center point and edge midpoints
Point(5) = {0.5, 0.5, 0, lc}; // Center
Point(6) = {0.5, 0, 0, lc};   // Midpoint-bottom
Point(7) = {1, 0.5, 0, lc};   // Midpoint-right
Point(8) = {0.5, 1, 0, lc};   // Midpoint-top
Point(9) = {0, 0.5, 0, lc};   // Midpoint-left

// Define the lines for the outer boundary segments
Line(1) = {1, 6}; // South-west
Line(2) = {6, 2}; // South-east
Line(3) = {2, 7}; // East-south
Line(4) = {7, 3}; // East-north
Line(5) = {3, 8}; // North-east
Line(6) = {8, 4}; // North-west
Line(7) = {4, 9}; // West-north
Line(8) = {9, 1}; // West-south

// Define the inner lines forming a cross through the center
Line(9)  = {6, 5};  // Bottom-center
Line(10) = {7, 5};  // Right-center
Line(11) = {8, 5};  // Top-center
Line(12) = {9, 5};  // Left-center

// Define the four square surfaces using curve loops
// Bottom-left square (Points: 1-6-5-9)
Curve Loop(1) = {1, 9, -12, 8};
Plane Surface(1) = {1};

// Bottom-right square (Points: 6-2-7-5)
Curve Loop(2) = {2, 3, 10, -9};
Plane Surface(2) = {2};

// Top-right square (Points: 5-7-3-8)
Curve Loop(3) = {11, -5, -4, 10};
Plane Surface(3) = {3};

// Top-left square (Points: 9-5-8-4)
Curve Loop(5) = {12, -11, 6, 7};
Plane Surface(5) = {5};

Transfinite Line {1, 2, 3, 4, 5, 6, 7, 8} = 2;
Transfinite Line {9, 10, 11, 12} = 20;


// --- 2. PHYSICAL GROUP DEFINITION ---

// Assign physical tags to the four surface areas
Physical Surface("area_bottom_left", 1) = {1};
Physical Surface("area_bottom_right", 2) = {2};
Physical Surface("area_top_right", 3) = {3};
Physical Surface("area_top_left", 4) = {5};

// Assign physical tags to the four outer boundaries
Physical Curve("south", 5) = {1, 2};
Physical Curve("east", 6) = {3, 4};
Physical Curve("north", 7) = {5, 6};
Physical Curve("west", 8) = {7, 8};
//+
Show "*";
