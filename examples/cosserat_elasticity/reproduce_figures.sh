#!/bin/bash


mkdir figures
echo " "
python graphical_geometrical_mesh_example_1_2.py
python graphical_geometrical_mesh_example_3.py
echo "Figure 1:: completed"

echo " "
python cosserat_error_plots.py
echo "Figure 2, 3, 4, 5, 7:: completed"

echo " "
python graphical_example_3_epsilon_jan.py
python graphical_example_3_couple_stress.py
echo "Figure 6:: completed"

echo " "
python cosserat_iteration_plots.py
echo "Figure 8, 9:: completed"



