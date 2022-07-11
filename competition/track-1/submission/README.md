# Submission
Once a model has been trained for `multi-scenario-v0` environments, place all necessary files to run the trained model inside this folder named `submission`. 

The files named `policy.py`, `requirements.txt`, and `explanation.md`, must be included with the submission. Its contents are explained below.

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
+ The dependencies will be installed prior to evaluating the submitted model.

## Explanation
+ Include an `explanation.md` file explaining the key techniques used in developing the submitted model.

## Submit to Codalab
+ Zip the `submission` folder. 
    + If the `submission` folder is located at `<path>/SMARTS/competition/submission`, then run `make submission.zip` from `<path>/SMARTS/competition` directory to easily create a zipped submission folder. 
+ Upload the `submission.zip` to CodaLab.
    + Go to the [CodaLab competition page](https://codalab.lisn.upsaclay.fr/).
    + Click `My Competitions -> Competitions I'm In`.
    + Select the SMARTS competition.
    + Click `Participate -> Submit/View Results -> Submit`
    + Upload the zipped submission folder.
