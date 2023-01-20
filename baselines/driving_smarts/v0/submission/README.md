# Submission
Once a model has been trained using `train/run.py`, save it into this directory. 

The files named `agent.py` and `requirements.txt`, must be included with the submission. Its contents are explained below.

## Agent
+ The directory must register the agent that is being requested in `__init__.py` or within `agent.py` which will be imported. Register the name appropriately to the module.
+ The implementation must include a `Agent` subclass which inherits from the `smarts.core.agent.Agent` class.
+ The `Agent` subclass must implement an `act` method which accepts observations and returns actions.
+ Any policy initialization, including loading of model may be performed inside the `__init__` method of the `Agent` subclass.
+ The `Agent.act(obs)` will be called during evaluation, with a multi-agent SMARTS observation as input and a agent action as the expected return value.
+ A random policy named `RandomAgent` (i.e. `"random-agent-v0"`) class is provided merely for reference.

## Wrappers
+ The file `policy.py` may include a `submitted_wrappers()` function.
+ The function `submitted_wrappers()` must return a list of callable wrappers, if any are used, else return an empty list `[]`. 
+ Use of wrappers is optional but not advised.
+ This option will be removed at a later date.

## Requirements
+ Create a `requirements.txt` file containing all the dependencies needed to run the submitted model. 
+ The dependencies will be installed prior to evaluating the submitted model.