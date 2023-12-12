Merge "network_3.brep";

///////////////////////////////////
// Physical geometry definitions //
///////////////////////////////////

// -- Physical entity definitions
Physical Surface(16) = {16};
Physical Surface(2) = {15};
Physical Surface(5) = {14};
Physical Surface(15) = {1};
Physical Surface(10) = {2};
Physical Surface(8) = {3};
Physical Surface(6) = {4};
Physical Surface(9) = {5};
Physical Surface(3) = {6};
Physical Surface(4) = {7};
Physical Surface(7) = {8};
Physical Surface(12) = {9};
Physical Surface(14) = {10};
Physical Surface(1) = {12};
Physical Surface(13) = {11};
Physical Surface(11) = {13};

// -- Physical entity intersection definitions
Physical Curve(1) = {22};

///////////////////////////
// Mesh size definitions //
///////////////////////////

// The order of definition is such that those geometries
// with the coarsest defined mesh sizes are placed first
// to ensure that for shared vertices the minimum mesh size is set

DefineConstant[ entityMeshSize_1 = 0.2 ];

// physical entity 11
Characteristic Length{ PointsOf{Surface{13};} } = entityMeshSize_1;
// physical entity 13
Characteristic Length{ PointsOf{Surface{11};} } = entityMeshSize_1;
// physical entity 1
Characteristic Length{ PointsOf{Surface{12};} } = entityMeshSize_1;
// physical entity 16
Characteristic Length{ PointsOf{Surface{16};} } = entityMeshSize_1;
// physical entity 3
Characteristic Length{ PointsOf{Surface{6};} } = entityMeshSize_1;
// physical entity 2
Characteristic Length{ PointsOf{Surface{15};} } = entityMeshSize_1;
// physical entity 15
Characteristic Length{ PointsOf{Surface{1};} } = entityMeshSize_1;
// physical entity 5
Characteristic Length{ PointsOf{Surface{14};} } = entityMeshSize_1;
// physical entity 10
Characteristic Length{ PointsOf{Surface{2};} } = entityMeshSize_1;
// physical entity 8
Characteristic Length{ PointsOf{Surface{3};} } = entityMeshSize_1;
// physical entity 6
Characteristic Length{ PointsOf{Surface{4};} } = entityMeshSize_1;
// physical entity 9
Characteristic Length{ PointsOf{Surface{5};} } = entityMeshSize_1;
// physical entity 4
Characteristic Length{ PointsOf{Surface{7};} } = entityMeshSize_1;
// physical entity 7
Characteristic Length{ PointsOf{Surface{8};} } = entityMeshSize_1;
// physical entity 12
Characteristic Length{ PointsOf{Surface{9};} } = entityMeshSize_1;
// physical entity 14
Characteristic Length{ PointsOf{Surface{10};} } = entityMeshSize_1;
