#!/bin/bash

echo "Computing numerical results:: started"
sh ./compute_results.sh
echo "Computing numerical results:: completed"

echo "Rendering paper figures:: started"
sh ./render_figures.sh
echo "Rendering paper figures:: completed"

tar -zcvf output_mfe.tar.gz output_example_1/ output_example_2/ output_example_3
tar -zcvf figures.tar.gz figures/