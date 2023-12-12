Merge "network_1.brep";

///////////////////////////////////
// Physical geometry definitions //
///////////////////////////////////

// -- Physical entity definitions
Physical Surface(17) = {34};
Physical Surface(7) = {14, 26};
Physical Surface(20) = {13, 33};
Physical Surface(4) = {10};
Physical Surface(10) = {9};
Physical Surface(25) = {8};
Physical Surface(1) = {6};
Physical Surface(30) = {21};
Physical Surface(6) = {7, 18};
Physical Surface(19) = {5};
Physical Surface(5) = {3};
Physical Surface(15) = {4};
Physical Surface(2) = {2};
Physical Surface(24) = {11};
Physical Surface(11) = {1, 12};
Physical Surface(13) = {15};
Physical Surface(27) = {16};
Physical Surface(22) = {17};
Physical Surface(14) = {19};
Physical Surface(3) = {20};
Physical Surface(18) = {22};
Physical Surface(21) = {23};
Physical Surface(26) = {24};
Physical Surface(9) = {25};
Physical Surface(28) = {27};
Physical Surface(12) = {28};
Physical Surface(8) = {29};
Physical Surface(29) = {30};
Physical Surface(16) = {31};
Physical Surface(23) = {32};

// -- Physical entity intersection definitions
Physical Curve(3) = {25};
Physical Curve(4) = {41};
Physical Curve(2) = {24};
Physical Curve(5) = {51};
Physical Curve(1) = {1};
Physical Curve(6) = {58};

///////////////////////////
// Mesh size definitions //
///////////////////////////

// The order of definition is such that those geometries
// with the coarsest defined mesh sizes are placed first
// to ensure that for shared vertices the minimum mesh size is set

DefineConstant[ entityMeshSize_1 = 0.2 ];

// physical entity 29
Characteristic Length{ PointsOf{Surface{30};} } = entityMeshSize_1;
// physical entity 8
Characteristic Length{ PointsOf{Surface{29};} } = entityMeshSize_1;
// physical entity 12
Characteristic Length{ PointsOf{Surface{28};} } = entityMeshSize_1;
// physical entity 28
Characteristic Length{ PointsOf{Surface{27};} } = entityMeshSize_1;
// physical entity 9
Characteristic Length{ PointsOf{Surface{25};} } = entityMeshSize_1;
// physical entity 26
Characteristic Length{ PointsOf{Surface{24};} } = entityMeshSize_1;
// physical entity 21
Characteristic Length{ PointsOf{Surface{23};} } = entityMeshSize_1;
// physical entity 18
Characteristic Length{ PointsOf{Surface{22};} } = entityMeshSize_1;
// physical entity 3
Characteristic Length{ PointsOf{Surface{20};} } = entityMeshSize_1;
// physical entity 14
Characteristic Length{ PointsOf{Surface{19};} } = entityMeshSize_1;
// physical entity 22
Characteristic Length{ PointsOf{Surface{17};} } = entityMeshSize_1;
// physical entity 27
Characteristic Length{ PointsOf{Surface{16};} } = entityMeshSize_1;
// physical entity 13
Characteristic Length{ PointsOf{Surface{15};} } = entityMeshSize_1;
// physical entity 11
Characteristic Length{ PointsOf{Surface{1, 12};} } = entityMeshSize_1;
// physical entity 16
Characteristic Length{ PointsOf{Surface{31};} } = entityMeshSize_1;
// physical entity 17
Characteristic Length{ PointsOf{Surface{34};} } = entityMeshSize_1;
// physical entity 4
Characteristic Length{ PointsOf{Surface{10};} } = entityMeshSize_1;
// physical entity 7
Characteristic Length{ PointsOf{Surface{14, 26};} } = entityMeshSize_1;
// physical entity 20
Characteristic Length{ PointsOf{Surface{13, 33};} } = entityMeshSize_1;
// physical entity 10
Characteristic Length{ PointsOf{Surface{9};} } = entityMeshSize_1;
// physical entity 25
Characteristic Length{ PointsOf{Surface{8};} } = entityMeshSize_1;
// physical entity 30
Characteristic Length{ PointsOf{Surface{21};} } = entityMeshSize_1;
// physical entity 1
Characteristic Length{ PointsOf{Surface{6};} } = entityMeshSize_1;
// physical entity 6
Characteristic Length{ PointsOf{Surface{7, 18};} } = entityMeshSize_1;
// physical entity 19
Characteristic Length{ PointsOf{Surface{5};} } = entityMeshSize_1;
// physical entity 5
Characteristic Length{ PointsOf{Surface{3};} } = entityMeshSize_1;
// physical entity 15
Characteristic Length{ PointsOf{Surface{4};} } = entityMeshSize_1;
// physical entity 2
Characteristic Length{ PointsOf{Surface{2};} } = entityMeshSize_1;
// physical entity 23
Characteristic Length{ PointsOf{Surface{32};} } = entityMeshSize_1;
// physical entity 24
Characteristic Length{ PointsOf{Surface{11};} } = entityMeshSize_1;
