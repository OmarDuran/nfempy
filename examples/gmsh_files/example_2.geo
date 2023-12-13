// Gmsh project created on Tue Dec 12 23:27:59 2023
SetFactory("OpenCASCADE");

lc = 0.05;
lci = 0.05;
hs = 0.5;
hi = 0.1;
l  = 1.0;

Point(1) = {-l, -l, 0, lc};
Point(2) = {-hs, -l, 0, lc};
Point(3) = {-hi, -l, 0, lci};
Point(5) = {+hi, -l, 0, lci};
Point(4) = {+hs, -l, 0, lc};
Point(6) = {+l, -l, 0, lc};

Point(7) = {+l, -hs, 0, lc};
Point(8) = {+l, -hi, 0, lci};
Point(9) = {+l, +hs, 0, lc};
Point(10) = {+l, +hi, 0, lci};

Point(11) = {+l, +l, 0, lc};
Point(12) = {+hi, +l, 0, lci};
Point(13) = {+hs, +l, 0, lc};
Point(14) = {-hi, +l, 0, lci};
Point(15) = {-hs, +l, 0, lc};

Point(16) = {-l, +l, 0, lc};
Point(17) = {-l, +hs, 0, lc};
Point(18) = {-l, +hi, 0, lci};
Point(19) = {-l, -hi, 0, lci};
Point(20) = {-l, -hs, 0, lc};


Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 5};
Line(4) = {5, 4};
Line(5) = {4, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 10};
Line(9) = {10, 9};
Line(10) = {9, 11};
Line(11) = {11, 13};
Line(12) = {13, 12};
Line(13) = {12, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};
Line(16) = {16, 17};
Line(17) = {17, 18};
Line(18) = {18, 19};
Line(19) = {19, 20};
Line(20) = {20, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
Plane Surface(1) = {1};


Line(21) = {2, 15};
Line(22) = {3, 14};
Line(23) = {5, 12};
Line(24) = {4, 13};
Line(25) = {20, 7};
Line(26) = {19, 8};
Line(27) = {18, 10};
Line(28) = {17, 9};

Line{21,22,23,24,25,26,27,28} In Surface{1};

Transfinite Line {21,22,23,24,25,26,27,28} = 100 Using Progression 1;



Physical Surface("domain", 1) = {1};
Physical Curve("south", 2) = {1, 2, 3, 4, 5};
Physical Curve("east", 3) = {6, 7, 8, 9, 10};
Physical Curve("north", 4) = {11, 12, 13, 14, 15};
Physical Curve("west", 5) = {16, 17, 18, 19, 20};
