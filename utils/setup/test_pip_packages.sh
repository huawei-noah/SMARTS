#!/usr/bin/env bash

venv_dir=`mktemp -u -d`
python3.7 -m venv ${venv_dir}
. ${venv_dir}/bin/activate

# Install our requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install our package on top
pip install .[test,train,camera-obs]
pip freeze | grep -v -E '^smarts' | grep -v 'pkg-resources==0.0.0' | diff -u requirements.txt -
