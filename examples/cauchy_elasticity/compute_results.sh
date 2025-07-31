#!/bin/bash


echo " "
echo "Example 1:: started"
python -OO mfe_elasticity_example_1.py
python -OO cg_elasticity_example_1.py
python -OO th_elasticity_example_1.py
mkdir output_example_1
mv ex_1* output_example_1/
mv geometric_* output_example_1/
echo "Example 1:: completed"


#echo " "
#echo "Example 2:: started"
#python -OO mfe_elasticity_example_2.py
#python -OO cg_elasticity_example_2.py
#python -OO th_elasticity_example_2.py
#mkdir output_example_2
#mv ex_2* output_example_2/
#mv geometric_* output_example_2/
#echo "Example 2:: completed"

tar -zcvf output_cauchy_fem.tar.gz output_example_1/ output_example_2/
