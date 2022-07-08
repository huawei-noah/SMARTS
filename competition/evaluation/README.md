# Evaluation
This folder contains the python script that CodaLab uses to evaluate the submissions.

## Local evaluation
+ To evaluate the trained model locally, do the following:
```bash
$ cd <path>/SMARTS/competition/evaluate
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ python3.8 evaluate.py --input_dir=<path/to/submission/folder> --local
# For example:
$ python3.8 evaluate.py --input_dir=<path>/SMARTS/competition/submission --local
```