#!/bin/bash

# compute figures
python error_plots.py
python graphical_example_3_l_2D.py

# convert figures
pdftops -eps figures/convergence_sigma_example_1.pdf figures/convergence_sigma_example_1.eps
pdftops -eps figures/convergence_sigma_example_2.pdf figures/convergence_sigma_example_2.eps
pdftops -eps figures/convergence_sigma_example_3.pdf figures/convergence_sigma_example_3.eps
pdftops -eps figures/convergence_u_example_1.pdf figures/convergence_u_example_1.eps
pdftops -eps figures/convergence_u_example_2.pdf figures/convergence_u_example_2.eps
pdftops -eps figures/convergence_u_example_3.pdf figures/convergence_u_example_3.eps
pdftops -eps figures/figure_spatial_ell.pdf figures/figure_spatial_ell.eps

rm -r figures/*.pdf