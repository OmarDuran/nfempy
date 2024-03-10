#!/bin/bash

echo "Computing numerical results:: started"
sh ./compute_results.sh
echo "Computing numerical results:: completed"

echo "Rendering paper figures:: started"
sh ./render_figures.sh
echo "Rendering paper figures:: completed"


