# Hosting A Competition For ULTRA In CodaLab

This folder contains files to organize a competition for Ultra in CodaLab. It contains two main directories:
- [`competition_bundle/`](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Competition-Bundle): Contains the files needed to create a CodaLab competition bundle.
  - [`scoring_program/`](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition): Contains the Python script that CodaLab will use to evaluate the agent submissions. 
- [`starting_kit`](https://github.com/codalab/codalab-competitions/wiki/User_Competition-Roadmap#creating-a-starting-kit): Contains all the files that participants need to install, build, train, and evaluate their submissions on test scenarios.

## Setup
For steps to install and run the starting-kit scripts, see [Setup](./starting_kit/README.md#Setup).

## Creating evaluation scenarios
Evaluation scenarios for Track 1 and Track 2 are defined by the configs in `competition_bundle/track1_evaluation_scenarios/` and `competition_bundle/track2_evaluation_scenarios/`, respectively. Generate the levels of these two tracks using the following commands:

```bash
# Generate Track 1 scenarios:
$ cd /path/to/SMARTS/ultra/competition
$ python starting_kit/scenarios/build_scenarios.py --task track1_evaluation_scenarios --level no-traffic-south-west --save-dir competition_bundle/track1_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
$ python starting_kit/scenarios/build_scenarios.py --task track1_evaluation_scenarios --level no-traffic-east-south --save-dir competition_bundle/track1_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/

# Generate Track 2 scenarios:
$ cd /path/to/SMARTS/ultra/competition
$ python starting_kit/scenarios/build_scenarios.py --task track2_evaluation_scenarios --level low-density --save-dir competition_bundle/track2_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
$ python starting_kit/scenarios/build_scenarios.py --task track2_evaluation_scenarios --level mid-density --save-dir competition_bundle/track2_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
$ python starting_kit/scenarios/build_scenarios.py --task track2_evaluation_scenarios --level high-density --save-dir competition_bundle/track2_evaluation_scenarios/ --root-dir competition_bundle --pool-dir starting_kit/scenarios/pool/
```

**NOTE:** If you use these evaluation scenarios (or any other configuration), the scoring function in `competition_bundle/scoring_program/evaluate.py` will likely have to be changed or tuned.

Alternatively, you can ask the SMARTS developers for evaluation scenarios that are currently held in a private repository. These private evaluation scenarios use the scoring function that is already implemented in `competition_bundle/scoring_program/evaluate.py`.

## Create the competition
1. Sign up for a CodaLab account on the [CodaLab site](https://codalab.org/).

2. Create the competition bundle:
    ```bash
    $ cd /path/to/SMARTS/ultra/competition
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

### CodaLab-like Evaluation
The first is how CodaLab would use the scoring program. CodaLab provides an input and output directory to the scoring program, and the scoring program then takes the submission and scenarios from the input directory, evaluates the submission, and outputs the scores in a text file in the output directory (see [here](https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition)).

To run the scoring program in this way on one of the baseline agents in the starting kit, the directory setup that CodaLab provides must be setup:

```bash
$ mkdir test_submission_dir/
$ mkdir test_submission_dir/input/
$ mkdir test_submission_dir/input/ref/
$ mkdir test_submission_dir/input/res/
$ mkdir test_submission_dir/output/
```

As outlined in the link above, the `input/ref/` directory contains all the data that the submission will be evaluated on. As an example, we can copy the Track 1 scenarios to this directory:

```bash
$ cp -r competition_bundle/track1_evaluation_scenarios/* test_submission_dir/input/ref/
```

As outlined in the link above, the `input/res/` directory contains all the data that for the submission. As an example, we can copy all the files the random baseline agent needs in order to run:

```bash
$ cp starting_kit/agents/random_baseline_agent/agent.py test_submission_dir/input/res/
```

Finally, we can run the scoring program, passing the input and output directory to the scoring program: 

```bash
$ python competition_bundle/scoring_program/evaluate.py codalab \
  --input-dir test_submission_dir/input/ \
  --output-dir test_submission_dir/output/
```

After evaluation is complete, `test_submission_dir/output/` should contain a `scores.txt`.

### Local Evaluation
Alternatively, the scoring program can be run in a more natural way where a submission directory, evaluation scenarios directory, and scores directory arguments provide directories for the scoring program to identify the submission, evaluation scenarios, and output directory, respectively:

```bash
$ python competition_bundle/scoring_program/evaluate.py local \
  --submission-dir starting_kit/agents/random_baseline_agent/ \
  --evaluation-sceanarios-dir ultra_competition/track1_evaluation_scenarios/ \
  --scores-dir ./example_scores
```

### Testing the Scripts in the Starting Kit
Follow the starting kit's [README.md](starting_kit/README.md) to test the scripts in the starting kit.