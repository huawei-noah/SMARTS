#!/usr/bin/env bash

pip install --upgrade pip
pip install -e .

dir="$PWD"
smartsdir="$(dirname "$dir")"

export PYTHONPATH=$smartsdir


