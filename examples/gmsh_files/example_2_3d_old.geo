// Gmsh project created on Tue Dec 12 23:27:59 2023
SetFactory("OpenCASCADE");

lcar1 = 0.15;

length = 1.0;
height = 1.0;
depth = 1.0;

xs = length/2;
ys = height/2;

Point(newp) = {length/2+xs,height/2+ys,depth,lcar1}; /* Point      1 */
Point(newp) = {length/2+xs,height/2+ys,0,lcar1}; /* Point      2 */
Point(newp) = {-length/2+xs,height/2+ys,depth,lcar1}; /* Point      3 */
Point(newp) = {-length/2+xs,-height/2+ys,depth,lcar1}; /* Point      4 */
Point(newp) = {length/2+xs,-height/2+ys,depth,lcar1}; /* Point      5 */
Point(newp) = {length/2+xs,-height/2+ys,0,lcar1}; /* Point      6 */
Point(newp) = {-length/2+xs,height/2+ys,0,lcar1}; /* Point      7 */
Point(newp) = {-length/2+xs,-height/2+ys,0,lcar1}; /* Point      8 */


Line(1) = {3,1};
Line(2) = {3,7};
Line(3) = {7,2};
Line(4) = {2,1};
Line(5) = {1,5};
Line(6) = {5,4};
Line(7) = {4,8};
Line(8) = {8,6};
Line(9) = {6,5};
Line(10) = {6,2};
Line(11) = {3,4};
Line(12) = {8,7};
Line Loop(13) = {-6,-5,-1,11};
Plane Surface(14) = {13};
Line Loop(15) = {4,5,-9,10};
Plane Surface(16) = {15};
Line Loop(17) = {-3,-12,8,10};
Plane Surface(18) = {17};
Line Loop(19) = {7,12,-2,11};
Plane Surface(20) = {19};
Line Loop(21) = {-4,-3,-2,1};
Plane Surface(22) = {21};
Line Loop(23) = {8,9,6,7};
Plane Surface(24) = {23};


Surface Loop(1) = {14, 24, 18, 22, 16, 20};
Volume(1) = {1};

Physical Volume("domain", 1) = {1};
Physical Surface("south", 2) = {14};
Physical Surface("east", 3) = {24};
Physical Surface("north", 4) = {18};
Physical Surface("west", 5) = {22};
Physical Surface("top", 6) = {16};
Physical Surface("bottom", 7) = {20};


