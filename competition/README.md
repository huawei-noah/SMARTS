# Hosting A Competition In CodaLab

This folder contains files to organize a competition for SMARTS in CodaLab. It contains two main directories:
- [`competition_bundle/`](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Competition-Bundle): Contains the files needed to create a CodaLab competition bundle.
  - [`scoring_program/`](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition): Contains the Python script that CodaLab will use to evaluate the agent submissions. 
- [`starting_kit`](https://github.com/codalab/codalab-competitions/wiki/User_Competition-Roadmap#creating-a-starting-kit): Contains all the files that participants need to install, build, train, and evaluate their submissions on test scenarios.

## Setup
For steps to install and run the starting-kit scripts, see [Setup](./starting_kit/README.md#Setup).

## Creating evaluation
Evaluation  for Track 1 and Track 2 are defined by the configs in `competition_bundle/track1_evaluation/` and `competition_bundle/track2_evaluation/`, respectively.

```bash
# Generate Track 1 scenarios:
$ cd /path/to/SMARTS/competition
$ python starting_kit/scenarios/build_scenarios.py --track competition_bundle/track1_evaluation/ --save-dir competition_bundle/track1_evaluation/
$ python starting_kit/scenarios/build_scenarios.py --track competition_bundle/track2_evaluation/ --save-dir competition_bundle/track2_evaluation/
```

**NOTE:** If you use these evaluation scenarios (or any other configuration), the scoring function in `competition_bundle/scoring_program/evaluate.py` will likely have to be changed or tuned.

## Create the competition
1. Sign up for a CodaLab account on the [CodaLab site](https://codalab.org/).

2. Create the competition bundle:
    ```bash
    $ cd /path/to/SMARTS/competition
    $ make competition_bundle.zip
    ```

3. Upload `competition_bundle.zip` to CodaLab.
    - Go to the [CodaLab competition page](https://competitions.codalab.org)
    - Click "My Competitions" at the top right
    - Click "Competitions I'm Running"
    - Click "Create Competition"
    - Upload the compressed competition bundle

4. Most aspects of the competition can then be edited on the CodaLab website. Notable parts of the competition that cannot be edited once the competition bundle is uploaded include the number of phases, and the number of leaderboard columns.

## Upload the starting kit
1. Create the starting kit:
    ```bash
    $ cd /path/to/SMARTS/ultra/competition
    $ make starting_kit.zip
    ```

2. Upload `starting_kit.zip` to the competition.
    - Go to your competition's page
    - Click "Edit"
    - Wait for page to fully load (takes ~ 2 minutes)
    - Once fully loaded, under one of the phases under the "Phases" header, click "My Datasets"
    - Upload the starting kit as a starting kit
    - Go back to the edit page and add the new starting kit as the starting kit for the desired phases

## Test the scoring program
The scoring program can be run in two ways.

### CodaLab-like evaluation
The first method is how CodaLab would use the scoring program. CodaLab provides an input and output directory to the scoring program, and the scoring program then takes the submission and scenarios from the input directory, evaluates the submission, and outputs the scores in a text file to the output directory (see [here](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition)).

To run the scoring program in this manner on one of the baseline agents in the starting kit, the directory structure that CodaLab provides must be setup.

```bash
$ cd /path/to/SMARTS/ultra/competition
$ mkdir test_submission_dir/
$ mkdir test_submission_dir/input/
$ mkdir test_submission_dir/input/ref/
$ mkdir test_submission_dir/input/res/
$ mkdir test_submission_dir/output/
```

The `input/ref/` directory contains all the data that the submission will be evaluated on. As an example, we can copy the Track 1 scenarios to this directory.

```bash
$ cd /path/to/SMARTS/ultra/competition
$ cp -r competition_bundle/track1_evaluation_scenarios/* test_submission_dir/input/ref/
```

The `input/res/` directory contains the submission data. As an example, we can copy files neede by the the random baseline agent.

```bash
$ cd /path/to/SMARTS/ultra/competition
$ cp starting_kit/agents/random_baseline_agent/agent.py test_submission_dir/input/res/
```

Finally, we can run the scoring program, by passing the input and output directory to the scoring program. 

```bash
$ cd /path/to/SMARTS/ultra/competition
$ python competition_bundle/scoring_program/evaluate.py codalab \
  --input-dir test_submission_dir/input/ \
  --output-dir test_submission_dir/output/
```

After evaluation is complete, `test_submission_dir/output/` should contain a `scores.txt`.

### Local evaluation
Alternatively, the scoring program can be run directly by providing a submission, evaluation-scenarios, and scores directory in the arguments. For details, see the "[submitting_an_agent_for_evaluation](./starting_kit/README.md#submitting-an-agent-for-evaluation)" section in the starting kit.
