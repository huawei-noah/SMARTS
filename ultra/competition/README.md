# Hosting A Competition For ULTRA In CodaLab

This folder contains files to organize a CodaLab competition for ULTRA. It contains two main directories:
- `competition_bundle/`: Contains the files needed to create a CodaLab competition
  bundle (see [here](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Competition-Bundle)).
  - `scoring_program/`: Contains the Python script that CodaLab will use to evaluate the agent submissions (see [here](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition)).
- `starting_kit`: Contains all the files that participants need to install all
  the dependencies, build and/or train their submissions, and evaluate them on test scenarios (see [here](https://github.com/codalab/codalab-competitions/wiki/User_Competition-Roadmap#creating-a-starting-kit)).

## Setup
For steps to install and run the evaluation and starting kit scripts, see [Setup](./starting_kit/README.md#Setup).

## Creating the evaluation scenarios
Evaluation scenarios for Track 1 and Track 2 are defined by the configs in `competition_bundle/track1_evaluation_scenarios/` and `competition_bundle/track2_evaluation_scenarios/`, respectively. Generate the levels of these two tracks using the following commands:

```bash
# Generate Track 1 scenarios:
$ python starting_kit/scenarios/build_scenarios.py --task track1_evaluation_scenarios --level no-traffic-south-west --save-dir competition_bundle/track1_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
$ python starting_kit/scenarios/build_scenarios.py --task track1_evaluation_scenarios --level no-traffic-east-south --save-dir competition_bundle/track1_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/

# Generate Track 2 scenarios:
$ python starting_kit/scenarios/build_scenarios.py --task track2_evaluation_scenarios --level low-density --save-dir competition_bundle/track2_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
$ python starting_kit/scenarios/build_scenarios.py --task track2_evaluation_scenarios --level mid-density --save-dir competition_bundle/track2_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
$ python starting_kit/scenarios/build_scenarios.py --task track2_evaluation_scenarios --level high-density --save-dir competition_bundle/track2_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
```

**NOTE:** If you use these evaluation scenarios (or any other configuration), the scoring function in `competition_bundle/scoring_program/evaluate.py` will likely have to be changed or tuned.

Alternatively, you can ask the SMARTS developers for evaluation scenarios that are currently held in a private repository. These private evaluation scenarios use the scoring function that is already implemented in
`competition_bundle/scoring_program/evaluate.py`.

## Creating the Competition

1. Sign up for a CodaLab account on the [CodaLab site](https://codalab.org/).

2. Create the competition bundle:

  ```bash
  $ make competition_bundle.zip
  ```

3. Upload `competition_bundle.zip` to CodaLab.
  - Go to the [CodaLab competition page](https://competitions.codalab.org)
  - Click "My Competitions" at the top right
  - Click "Competitions I'm Running"
  - Click "Create Competition"
  - Upload the compressed competition bundle

**NOTE:** Most aspects of the competition can then be edited on the CodaLab website.
Noteable parts of the competition that cannot be edited once the competition bundle is
uploaded include the number phases, and the number of leaderboard columns.

## Uploading the Starting Kit

1. Create the starting kit:

  ```bash
  $ make starting_kit.zip
  ```

2. Upload `starting_kit.zip` to the competition.
  - Go to your competition's page
  - Click "Edit"
  - Wait for page to fully load (takes ~ 2 minutes)
  - Once fully loaded, under one of the phases under the "Phases" header, click
    "My Datasets"
  - Upload the starting kit as a starting kit
  - Go back to the edit page and add the new starting kit as the starting kit for the
    desired phases

## Testing the Scoring Program

The scoring program can be run in two ways.

### CodaLab-like Evaluation

The first is how CodaLab would use the scoring program. CodaLab provides an input and
output directory to the scoring program, and the scoring program then takes the
submission and scenarios from the input directory, evaluates the submission, and outputs
the scores in a text file in the output directory
(see [here](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition)).

To run the scoring program in this way on one of the baseline agents in the starting
kit, the directory setup that CodaLab provides must be setup:

```bash
$ mkdir test_submission_dir/
$ mkdir test_submission_dir/input/
$ mkdir test_submission_dir/input/ref/
$ mkdir test_submission_dir/input/res/
$ mkdir test_submission_dir/output/
```

As outlined in the link above, the `input/ref/` directory contains all the data that the
submission will be evaluated on. As an example, we can copy the Track 1 scenarios to
this directory:

```bash
$ cp -r competition_bundle/track1_evaluation_scenarios/* test_submission_dir/input/ref/
```

As outlined in the link above, the `input/res/` directory contains all the data that for
the submission. As an example, we can copy all the files the random baseline agent needs
in order to run:

```bash
$ cp starting_kit/agents/random_baseline_agent/agent.py test_submission_dir/input/res/
```

Finally, we can run the scoring program, passing the input and output directory to the
scoring program: 

```bash
$ python competition_bundle/scoring_program/evaluate.py codalab \
  --input-dir test_submission_dir/input/
  --output-dir test_submission_dir/output/
```

After evaluation is complete, `test_submission_dir/output/` should contain a
`scores.txt`.

### Local Evaluation

Alternatively, the scoring program can be run in a more natural way where a submission
directory, evaluation scenarios directory, and scores directory arguments provide
directories for the scoring program to identify the submission, evaluation scenarios,
and output directory, respectively:

```bash
$ python competition_bundle/scoring_program/evaluate.py local \
  --submission-dir starting_kit/agents/random_baseline_agent/
  --evaluation-sceanarios-dir ultra_2021_competition/track1_evaluation_scenarios/
  --scores-dir ./exxample_scores
```

### Testing the Scripts in the Starting Kit

Follow the starting kit's [README.md](starting_kit/README.md) to test the scripts in the
starting kit.

## TODOs

- [x] Update baseline agents in the starting kit with the newest code
  - This can be done once there are no more changes to the baseline's code in ULTRA
- [x] Take a look at starting kit's install_deps.sh... Do we need the X11 stuff?
  - Specifically lines 56 and 57
  - This is a very low priority task
  - Edit: We should keep lines 56 and 57.
- [x] Add contents to starting kit's Dockerfile
- [x] Create an actual ULTRA wheel and replace the placeholder in the starting kit
- [ ] Add an ULTRA wheel in the starting kit
  - This can be done once an ULTRA version is finalized
  - See the SMARTS competition archive's starter kit for an example of what we had in
    mind (https://rnd-gitlab-ca.huawei.com/smarts/smarts-codalab-archive/-/tree/master/starter_kit)
- [ ] Ensure the Dockerfile in the starting kit is pulling from the correct ULTRA branch
- [x] Specify what has to be used for agents (e.g. action and observation space,
  NeighborhoodVehicles, and Waypoints) in the starting kit's README instructions
- [x] Specify what a submission should look like (`agent.py` file with `agent_spec`
  variable) in the starting kit's README instructions
- [x] Allow option to specify headless mode or not in starting kit's run.py
- [x] Put envision instructions in the starting kit
- [x] Should the `build_scenarios.py` require the directory that holds the `config.yaml`
  start with `task`? (E.g. `taskX`, `task1`)
  - Currently, the `--task` argument is the name of the directory containing the
    `config.yaml`; if it was like this in ULTRA and we wanted to build Task 1, we would
    have to specify `--task task1` instead of `--task 1`
  - The way it is now (where we specify the whole folder name) might be simpler for
    participants
  - Again a fairly minor task
  - IMPLEMENTED SOLUTION: The directory does not need to start with "task".
- [x] Print which scenario is running for each episode in the `run.py` scripts
  - This can be added once an ULTRA version with [#1031](https://github.com/huawei-noah/SMARTS/pull/1031)
    is released
- [ ] Add evaluation script to starting kit
  - This can be done once the evaluation script is complete
  - Create a new directory `evaluation/` under `starting_kit/` and copy
    `competition_bundle/scoring_program/evaluate.py` to `starting_kit/evaluation/` so
    that the instructions for evaluation in `starting_kit/README.md` are consistent
- [x] Add instructions on how to use the baselines (the `checkpoint_dir` argument)
- [x] Use radius 200 for `NeighborhoodVehicles`, not 100.
- [x] Add DoneCriteria documentation to the starting kit's README
  - Include what the DoneCriteria is
  - Include what each value of the DoneCriteria means
  - Include what type of DoneCriteria we will use to evaluate them
- [x] If it doesn't install on macOS, update the setup instructions in the starting
  kit's README.
  - As of August 19, 2021, the SMARTS version we are planning to use to support this
    competition does not support macOS, therefore only Linux will be supported for
    native installation. Other operating systems can use Docker.
- [ ] Explain how trained the RL baseline networks in the starting kit (the neural
  network weights that come with the SAC and PPO agents) are and where they came from
  - This can be completed once we have baselines models
  - In order to obtain baseline models, it would be ideal if we could have a final
    SMARTS version to train them in
- [x] Explicitly define the RL baseline's agent interface
- [x] Update `starting_kit/install_deps.sh`
