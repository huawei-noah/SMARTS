#!/usr/bin/env bash

# Installs SMARTS/ULTRA dependencies
pip install --upgrade pip
pip install -e .

# Initializes the path to the SMARTS repo
dir="$PWD"
path_to_smarts_dir="$(dirname "$dir")"
export_command=$'export PYTHONPATH='\"'${PYTHONPATH}:'$path_to_smarts_dir'"'

# Checks whether .bashrc script is updated. If not, new path is added
if grep -q 'PYTHONPATH='\"'${PYTHONPATH}:'$path_to_smarts_dir'"' ~/.bashrc; then
    echo "Path already exists in PYTHONPATH"
else
    echo '# Add absolute path of SMARTS repo to PYTHONPATH' >> ~/.bashrc
    echo $export_command >> ~/.bashrc 
fi
