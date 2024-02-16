
lc = 0.5;
r = 1.0;
h = 1.0;

Point(1) = {0, 0, 0, lc};
Point(2) = {+r, 0.0, 0, lc};
Point(3) = {0.0, +r, 0, lc};
Point(4) = {-r, 0.0, 0, lc};
Point(5) = {0.0, -r, 0, lc};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};


Extrude {0, 0, h} {
  Surface{1}; Curve{1}; Curve{2}; Curve{3}; Curve{4}; 
}


Physical Volume("domain", 1) = {1};
Physical Surface("bc_east", 2) = {13};
Physical Surface("bc_south", 3) = {17};
Physical Surface("bc_west", 4) = {21};
Physical Surface("bc_north", 5) = {25};
Physical Surface("bc_bottom", 6) = {1};
Physical Surface("bc_top", 7) = {26};
