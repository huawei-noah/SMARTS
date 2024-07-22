# CRITERIA

This repository contains code for computation of metrics presented in [C. Chen, M. Pourkeshavarz, A. Rasouli, CRITERIA: a New Benchmarking Paradigm for Evaluating Trajectory Prediction Models for Autonomous Driving, ICRA, 2024](https://arxiv.org/pdf/2310.07794)

## Cloning

```
# Clone this particular branch
$ git clone -b CRITERIA-latest https://github.com/huawei-noah/smarts.git CRITERIA
```

## Installation

This project is installable as a package named `CRITERIA`.

If you have argoverse already installed it should be as easy as:
```bash
# use -e for editable if you need to
$ pip install .
```

## Argoverse

Follow the instructions here (if you have an install error see below): https://github.com/argoverse/argoverse-api/blob/master/README.md

### Argoverse dependency errors
We provide an option that will install `argoverse` dependencies and bypass the `sklearn` installation error if those occur.

```bash
# Install argoverse without dependencies
pip install --no-deps ./argoverse-api
# Install working argoverse dependencies (as of writing)
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -e .[argoverse_dependencies]
```

If you wish to solve dependency issues directly the following dependencies are known to be trouble:

```requirements.txt
numpy==1.9 # ~=1.9.3 instead will work
sklearn    # package is deprecated
```

### Simplified links

It is suggested that the repository is set up this way:


```sh
CRITERIA
    |-- argoverse_motion_forecasting # can be soft link
    |    -- val
    |-- predictions # can be soft link
    |   |-- HiVT
    |   |-- mmTransformer
    |   |-- LaneGCNpppp
    |   |-- TNT
    |    -- FTGN
    |-- LICENSE
    |-- README.md
    |-- examples # dir
    |-- pyproject.toml
    |-- setup.py
     -- src # dir
```


## Use

For any of the utilities that are part of the package they can be run similarly to the following:

```bash
$ python -m CRITERIA.compute_metrics.preprocess_argoverse
```

The repository version contains an `examples` directory that has additional scripts. Some setup is needed first.

Use in this order:

```bash
# Must be run.
python -m CRITERIA.compute_metrics.preprocess_argoverse # generates vis_map
# May need to be run
python -m CRITERIA.compute_metrics.compute_metric_result --TNT --LaneGCN --HiVT --FTGN --mmTransformer
# May need to be run, requires all metrics above
python -m CRITERIA.compute_metrics.group_result # generates models_metrics_results.pkl
# Requires all above
python -m CRITERIA.scene_classification.scenario_difficulty_classify
python -m CRITERIA.scene_classification.scenario_distance_classify
python -m CRITERIA.scene_classification.scenario_intersect_classify
# Requires all above
python -m CRITERIA.scene_classification.group_classify # generates 12_scenes.xlsx
# Requires group_classify
python -m CRITERIA.scene_classification.scenes_result
python -m CRITERIA.scene_classification.weighted_difficulty_result
# Any scripts in examples
python examples/*
```


## FAQ

### How do I change the script configuration?

The package contains a `CRITERIA.toml` configuration file that is used by default. You can find this file using the following command:

```bash
$ python -m CRITERIA.resources.list
# output
.../CRITERIA.resources/CRITERIA.toml
# assume some editor like vim, nano, .etc
editor `python -m CRITERIA.resources.list`
```

To use a different configuration instead you can use the `-c` option on any of the executable scripts.

<a name="citation"></a>

## Citation

If you use the code, please cite:

```
@InProceedings{Chen_2024_ICRA,
author = {Chen, Changhe and Pourkeshavarz, Mozhgan and Rasouli, Amir},
title = {CRITERIA: a New Benchmarking Paradigm for Evaluating Trajectory Prediction Models for Autonomous Driving},
booktitle = {International Conference on Robotics and Automation (ICRA), year = {2024}}
}
```

