# Submission
Once a model has been trained offline, place all necessary files to run the saved model for inference inside this folder named `submission`. 

The files named `policy.py`, `requirements.txt`, and `explanation.md`, must be included within this folder. Its contents are identical to that of Track-1 and they are explained at 
+ [Policy](../../track1/submission/README.md#Policy)
+ [Wrappers](../../track1/submission/README.md#Wrappers)
+ [Requirements](../../track1/submission/README.md#Requirements)
+ [Explanation](../../track1/submission/README.md#Explanation)

## Submit to Codalab
+ Zip the entire `track2` folder. 
    + If the `track2` folder is located at `<path>/SMARTS/competition/track2`, then run the following to easily create a zipped folder. 
        ```bash
        $ cd <path>/SMARTS/competition
        $ make track2_submission.zip 
        ```
+ Upload the `track2.zip` to CodaLab.
    + Go to the [CodaLab competition page](https://codalab.lisn.upsaclay.fr/).
    + Click `My Competitions -> Competitions I'm In`.
    + Select the SMARTS competition.
    + Click `Participate -> Submit/View Results -> Submit`
    + Upload the zipped folder.

