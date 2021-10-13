#!/usr/bin/env bash

venv_dir=`mktemp -u -d`
python3.7 -m venv ${venv_dir}
. ${venv_dir}/bin/activate
pip install --upgrade pip

# Install packages
pip install .[test,train,camera-obs]

# Compare with requirements.txt
pip freeze | grep -v -E '^smarts' | grep -v 'pkg-resources==0.0.0' | diff -u requirements.txt -
