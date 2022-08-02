#!/bin/sh
# entrypoint.sh
# [Do Not Modify]
python3.8 /SMARTS/competition/track2/train/train.py
zip -j -r /SMARTS/competition/output/train.zip /SMARTS/competition/track2/train
zip -j -r /SMARTS/competition/output/submission.zip /SMARTS/competition/track2/submission
