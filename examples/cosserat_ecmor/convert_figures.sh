#!/bin/bash

pdftops -eps figures/convergence_sigma_example_1.pdf figures/convergence_sigma_example_1.eps
pdftops -eps figures/convergence_sigma_example_2.pdf figures/convergence_sigma_example_2.eps
pdftops -eps figures/convergence_sigma_example_3.pdf figures/convergence_sigma_example_3.eps
pdftops -eps figures/convergence_u_example_1.pdf figures/convergence_u_example_1.eps
pdftops -eps figures/convergence_u_example_2.pdf figures/convergence_u_example_2.eps
pdftops -eps figures/convergence_u_example_3.pdf figures/convergence_u_example_3.eps

rm -r figures/*.pdf