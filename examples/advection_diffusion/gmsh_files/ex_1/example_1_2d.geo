SetFactory("OpenCASCADE");

// ref levels
//s=2;
//s=4;
//s=8;
//s=16;
//s=32;

s=4;
lc = 1.0/s;

// inner box
Point(1) = {0, 0, 0, lc};
Point(2) = {10, 0, 0, lc};
Point(3) = {10, 1, 0, lc};
Point(4) = {0, 1, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};


Physical Surface("domain", 1) = {1};
Physical Line("south", 2) = {1};
Physical Line("east", 3) = {2};
Physical Line("north", 4) = {3};
Physical Line("west", 5) = {4};
