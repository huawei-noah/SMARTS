#!/bin/sh
# entrypoint.sh
# [Do Not Modify]
python3.8 /track2/train/train.py
cp -r /track2/* /output