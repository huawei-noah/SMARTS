#!/bin/sh
# entrypoint.sh
# [Do Not Modify]
python3.8 /SMARTS/competition/track2/train/train.py
cd /SMARTS/competition/track2
zip -r /SMARTS/competition/output/train.zip ./train
zip -r /SMARTS/competition/output/submission.zip ./submission
cd -