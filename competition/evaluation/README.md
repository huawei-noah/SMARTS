# Evaluation
This folder contains the python script that CodaLab uses to evaluate the submissions.

## Test cases for Track-1
+ Submitted models will be evaluated on scenarios similar to those given for training, namely, intersection, merging, cruising, and cut-in scenarios.

## Score for Track-1
+ Submitted models are scored on four aspects, namely,
    + Completion: Number of goals completed.
    + Time: Number of steps taken to complete the scenarios.
    + Humanness: Similarity to human behaviour.
    + Rules: Compliance with traffic rules.
+ Factors included in `Humanness` aspect, include 
    + distance to obstacles
    + jerk
    + lane center offset
    + steering rate
    + velocity offset
    + yaw rate
+ Factors included in the `Rules` aspect, include events such as 
    + collisions
    + driving off road
    + driving off route 
    + driving on shoulder
    + driving wrong way
+ Each score component must be minimized. The lower the value, the better it is.
+ Overall rank is obtained by sorting each score component in ascending order, with a priority order of Completion > Time > Humanness > Rules .

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