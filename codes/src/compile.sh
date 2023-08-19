#!/bin/bash

python setup.py install

# after installation, delete the temporary files in the folder
rm -rf ./build
rm -rf ./dist
find . -type d -name "*.egg-info" -exec rm -rf {} +


# fix import issue of fixmesh
python edit_fixmesh.py

