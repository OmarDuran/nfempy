// Gmsh project created on Tue Dec 12 23:27:59 2023
SetFactory("OpenCASCADE");


lc = 0.1;
lci = 0.1;
hi = 1.0/3.0;
hs = 2.0/3.0;
l  = 1.0;

Point(1) = {-1, -1, 0, lc};
Point(2) = {+hi, -1.0, 0, lc};
Point(3) = {+hs, -1, 0, lc};
Point(4) = {+1, -1, 0, lc};
Point(5) = {+1, +1, 0, lc};
Point(6) = {-1, +1, 0, lc};
Point(7) = {-1, +hs, 0, lc};
Point(8) = {-1, +hi, 0, lc};

Point(9) = {+hi, +hi, 0, lc};
Point(10) = {+hs, +hs, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};


Line(9) = {2, 9};
Line(10) = {8, 9};
Line(11) = {3, 10};
Line(12) = {7, 10};
Line(13) = {9, 10};


Line{9,10,11,12,13} In Surface{1};

Transfinite Line {9,10,11,12,13} = 10 Using Progression 1;
Transfinite Line {13} = 5 Using Progression 1;


Physical Surface("domain", 1) = {1};
Physical Curve("south", 2) = {1, 2, 3};
Physical Curve("east", 3) = {4};
Physical Curve("north", 4) = {5};
Physical Curve("west", 5) = {6, 7, 8};


