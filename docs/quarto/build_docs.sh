#!/bin/bash

# Build both PDF and HTML versions
quarto render simulation_documentation.qmd --to pdf
quarto render simulation_documentation.qmd --to html

# Copy the generated files to their respective locations
mkdir -p ../static
cp simulation_documentation.pdf ../static/
cp simulation_documentation.html ../static/

# Copy to Shiny app's static directory
mkdir -p ../../src/app/static
cp simulation_documentation.pdf ../../src/app/static/
cp simulation_documentation.html ../../src/app/static/

echo "Documentation built and copied to static directories" 