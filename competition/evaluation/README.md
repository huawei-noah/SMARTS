# Evaluation
This folder contains the python script that CodaLab uses to evaluate the submissions.

## Test cases
+ Submitted models will be evaluated on scenarios similar to those given for training and on additional hidden scenarios. Test scenarios include intersections, merging, cruising, cut-ins, curved roads, etc.

## Score
+ Submitted models are scored on four aspects, namely,
    + Completion: Number of goals completed.
    + Time: Number of steps taken and the final distance to goal.
    + Humanness: Similarity to human behaviour.
    + Rules: Compliance with traffic rules.
+ Factors included in `Humanness` aspect, include 
    + distance to obstacles
    + angular jerk
    + linear jerk
    + lane center offset
+ Factors included in the `Rules` aspect, include events such as 
    + exceeding speed limit
    + driving wrong way
+ Each score component must be minimized. The lower the value, the better it is.
+ Overall rank is obtained by sorting each score component in ascending order, with a priority order of Completion > Time > Humanness > Rules .

## Setup
1. Install [git lfs](https://git-lfs.github.com/).
    ```bash
    $ sudo apt-get install git-lfs
    ```

## Local evaluation
+ To evaluate the trained model locally, do the following:
    ```bash
    $ cd <path>/SMARTS/competition/evaluation
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip setuptools wheel
    $ python3.8 evaluate.py --input_dir=<path/to/submission/folder> --output_dir=<path/to/output/folder> --auto_install_pip_deps --local 
    # Evaluation scores will be written to the output_dir folder.
    # For example:
    $ python3.8 evaluate.py --input_dir=<path>/SMARTS/competition/track1/submission --output_dir=../output --auto_install_pip_deps --local
    ```