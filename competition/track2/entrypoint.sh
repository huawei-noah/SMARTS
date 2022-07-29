#!/bin/sh
# entrypoint.sh
# [Do Not Modify]
python3.8 /track2/train/train.py
zip -j -r /output/train.zip /track2/train
zip -j -r /output/submission.zip /track2/submission