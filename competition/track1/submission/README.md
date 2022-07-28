# Submission
Once an RL model has been trained, save it into the `track1/submission` folder. Place all necessary files to run the saved model for inference inside the `track1/submission` folder.

Besides the saved RL model, the files named `policy.py`, `requirements.txt`, and `explanation.md`, must be included with the submission. Its contents are explained below.

## Policy
+ The file `policy.py` must include a `Policy` class which inherits from the `BasePolicy` class.
+ The `Policy` class must implement an `act` method which accepts observations and returns actions.
+ Any policy initialization, including loading of model may be performed inside the `__init__` method of the `Policy` class.
+ The `submission/policy.py::Policy.act(obs)` will be called during evaluation, with a multi-agent SMARTS observation as input and a multi-agent action as the expected return value.
+ A random policy named `RandomPolicy` class is provided merely for reference.

## Wrappers
+ The file `policy.py` must include a `submitted_wrappers()` function.
+ The function `submitted_wrappers()` must return a list of callable wrappers, if any are used, else return an empty list `[]`. 
+ Use of wrappers is optional.

## Requirements
+ Create a `requirements.txt` file containing all the dependencies needed to run the submitted model. 
+ The dependencies will be installed prior to evaluating the submitted model.

## Explanation
+ Include an `explanation.md` file explaining the key techniques used in developing the submitted model.
+ Provide a link to your source code, preferably in GitHub.

## Submit to Codalab
+ Zip the `submission` folder. 
    + If the `submission` folder is located at `<path>/SMARTS/competition/track1/submission`, then run the following to easily create a zipped submission folder. 
        ```bash
        $ cd <path>/SMARTS/competition
        $ make track1_submission.zip 
        ```
+ Upload the `track1_submission.zip` to CodaLab.
    + Go to the [CodaLab competition page](https://codalab.lisn.upsaclay.fr/).
    + Click `My Competitions -> Competitions I'm In`.
    + Select the SMARTS competition.
    + Click `Participate -> Submit/View Results -> Submit`
    + Upload the zipped submission folder.
