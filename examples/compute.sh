#!/bin/bash

mkdir output_example_1
mkdir output_example_2

python -OO mfe_cosserat_elasticity_example_1.py
mv m1_* output_example_1/
mv m2_* output_example_1/
mv m3_* output_example_1/
mv geometric_* output_example_1/

python -OO mfe_cosserat_elasticity_example_2.py
mv m1_* output_example_2/
mv m2_* output_example_2/
mv m3_* output_example_2/
mv geometric_* output_example_2/

