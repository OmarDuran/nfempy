// Gmsh project created on Tue Dec 12 23:27:59 2023
SetFactory("OpenCASCADE");

// ref levels
//s=2;
//s=5;
//s=9;
//s=18;

s=2;

lc = 1.0/s;
hi = 1.0/3.0;
hs = 2.0/3.0;
ho = 1.0;

// inner box
pi1=newp; Point(pi1) = {0, 0, 0, lc};
pi2=newp; Point(pi2) = {+hi, 0, 0, lc};
pi3=newp; Point(pi3) = {+hi, +hi, 0, lc};
pi4=newp; Point(pi4) = {0, +hi, 0, lc};

pi5=newp; Point(pi5) = {0, 0, +hi, lc};
pi6=newp; Point(pi6) = {+hi, 0, +hi, lc};
pi7=newp; Point(pi7) = {+hi, +hi, +hi, lc};
pi8=newp; Point(pi8) = {0, +hi, +hi, lc};


// mid box
pm1=newp; Point(pm1) = {0, 0, 0, lc};
pm2=newp; Point(pm2) = {+hs, 0, 0, lc};
pm3=newp; Point(pm3) = {+hs, +hs, 0, lc};
pm4=newp; Point(pm4) = {0, +hs, 0, lc};

pm5=newp; Point(pm5) = {0, 0, +hs, lc};
pm6=newp; Point(pm6) = {+hs, 0, +hs, lc};
pm7=newp; Point(pm7) = {+hs, +hs, +hs, lc};
pm8=newp; Point(pm8) = {0, +hs, +hs, lc};


// outer box
po1=newp; Point(po1) = {0, 0, 0, lc};
po2=newp; Point(po2) = {+ho, 0, 0, lc};
po3=newp; Point(po3) = {+ho, +ho, 0, lc};
po4=newp; Point(po4) = {0, +ho, 0, lc};

po5=newp; Point(po5) = {0, 0, +ho, lc};
po6=newp; Point(po6) = {+ho, 0, +ho, lc};
po7=newp; Point(po7) = {+ho, +ho, +ho, lc};
po8=newp; Point(po8) = {0, +ho, +ho, lc};




Line(1) = {1, 2};
Line(2) = {2, 10};
Line(3) = {10, 18};
Line(4) = {18, 19};
Line(5) = {19, 20};
Line(6) = {20, 12};
Line(7) = {12, 4};
Line(8) = {4, 1};
Line(9) = {18, 22};
Line(10) = {19, 23};
Line(11) = {20, 24};
Line(12) = {1, 5};
Line(13) = {5, 13};
Line(14) = {13, 21};
Line(15) = {21, 22};
Line(16) = {21, 24};
Line(17) = {22, 23};
Line(18) = {23, 24};
Line(19) = {2, 3};
Line(20) = {3, 7};
Line(21) = {2, 6};
Line(22) = {6, 7};
Line(23) = {3, 4};
Line(24) = {4, 8};
Line(25) = {7, 8};
Line(26) = {5, 6};
Line(27) = {5, 8};
Line(28) = {10, 11};
Line(29) = {10, 14};
Line(30) = {11, 15};
Line(31) = {14, 15};
Line(32) = {13, 14};
Line(33) = {16, 15};
Line(34) = {13, 16};
Line(35) = {16, 12};
Line(36) = {12, 11};
Curve Loop(1) = {1, 21, -26, -12};
Plane Surface(1) = {1};
Curve Loop(2) = {13, 32, -29, -2, 21, -26};
Plane Surface(2) = {2};
Curve Loop(3) = {29, -32, 14, 15, -9, -3};
Plane Surface(3) = {3};
Curve Loop(4) = {4, 10, -17, -9};
Plane Surface(4) = {4};
Curve Loop(5) = {5, 11, -18, -10};
Plane Surface(5) = {5};
Curve Loop(6) = {1, 19, 23, 8};
Plane Surface(6) = {6};
Curve Loop(7) = {2, 28, -36, 7, -23, -19};
Plane Surface(7) = {7};
Curve Loop(8) = {3, 4, 5, 6, 36, -28};
Plane Surface(8) = {8};
Curve Loop(9) = {8, 12, 27, -24};
Plane Surface(9) = {9};
Curve Loop(10) = {27, -24, -7, -35, -34, -13};
Plane Surface(10) = {10};
Curve Loop(11) = {6, -35, -34, 14, 16, -11};
Plane Surface(11) = {11};
Curve Loop(12) = {15, 17, 18, -16};
Plane Surface(12) = {12};
Curve Loop(13) = {22, 25, -27, 26};
Plane Surface(13) = {13};
Curve Loop(14) = {21, 22, -20, -19};
Plane Surface(14) = {14};
Curve Loop(15) = {24, -25, -20, 23};
Plane Surface(15) = {15};
Curve Loop(16) = {29, 31, -30, -28};
Plane Surface(16) = {16};
Curve Loop(17) = {36, 30, -33, 35};
Plane Surface(17) = {17};
Curve Loop(18) = {34, 33, -31, -32};
Plane Surface(18) = {18};

Line(37) = {7, 15};
Line(38) = {8, 16};
Line(39) = {3, 11};
Curve Loop(19) = {38, 33, -37, 25};
Plane Surface(19) = {19};
Curve Loop(20) = {39, 30, -37, -20};
Plane Surface(20) = {20};
Line(40) = {6, 14};
Curve Loop(21) = {37, -31, -40, 22};
Plane Surface(21) = {21};


Surface Loop(1) = {4, 8, 3, 11, 12, 5, 2, 10, 7, 1, 6, 9};
Volume(1) = {1};


Line{38} In Surface{10};
Line{39} In Surface{7};
Line{40} In Surface{2};
Surface{13,14,15,16,17,18,19,20,21} In Volume{1};

Physical Volume("domain", 1) = {1};
Physical Surface("south", 2) = {1,2,3};
Physical Surface("east", 3) = {4};
Physical Surface("north", 4) = {5};
Physical Surface("west", 5) = {9,10,11};
Physical Surface("top", 6) = {12};
Physical Surface("bottom", 7) = {6,7,8};
