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
+ There is a utility for setting up the base dependencies.
+ Be aware that you will need [git lfs](git-lfs.github.com) in order to get all dependencies.

### Option 1
+ Install dependencies using the utility script.

    ```bash
    # Use --no-cache if you wish a full reinstall
    $ python -m auto_install
    ```
### Option 2
+ The alternative is to install from source. This allows for updates of the source.

    ```bash
    $ cd <to/your/repo/storage>
    $ git clone <smarts_repo> # likely already located at ../.. from this file
    $ pip install -e <smarts_repo>
    # check that git lfs is installed:
    $ git lfs --version
    $ git clone https://malban:ATBByMTp2W2MnVGsxHBYwEbNsVca00608BD5@bitbucket.org/malban/bubble_env.git
    $ pip install -e ./bubble_env
    ```

## Local evaluation
+ To evaluate the trained model locally, do the following:
    ```bash
    $ cd <path>/SMARTS/competition/evaluation
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ python3.8 evaluate.py --input_dir=<path/to/submission/folder> --output_dir=<path/to/output/folder> --local
    # Evaluation scores will be written to the output_dir folder.
    # For example:
    $ python3.8 evaluate.py --input_dir=<path>/SMARTS/competition/track1/submission --output_dir=../output --local
    ```