#!/usr/bin/env bash

venv_dir=`mktemp -u -d`
python3.7 -m venv ${venv_dir}
. ${venv_dir}/bin/activate

# install our requirements
pip install --upgrade pip
pip install -r requirements.txt

# install our package on top
pip install .
pip install .[train]
pip freeze | grep -v -E '^smarts' | grep -v 'pkg-resources==0.0.0' | diff -u requirements.txt -
