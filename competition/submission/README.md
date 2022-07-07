# Submission
These instructions only pertain to the NeurIPS 2022 competition.

Once a model has been trained for `multi-scenario-v0` environments, place all necessary files to run the trained model inside this folder named `submission`. 

The files named `policy.py` and `requirements.txt` must be included with the submission. Its contents are explained below.

## Policy
+ The file `policy.py` must include a `Policy` class which inherits from the `BasePolicy` class.
+ The `Policy` class must implement an `act` method which accepts observations and returns actions.
+ Any policy initialization, including loading of model may be performed inside the `__init__` method of the `Policy` class.
+ A random policy named `RandomPolicy` class is provided for reference.

## Wrappers
+ The file `policy.py` must include a `submitted_wrappers()` function.
+ The function `submitted_wrappers()` must return a list of callable wrappers, if any are used, else return an empty list `[]`. 
+ Use of wrappers is optional.

## Requirements
+ Create a `requirements.txt` file containing all the dependencies needed to run the submitted model. 
+ The dependencies will be installed prior to evaluating the submitted code.