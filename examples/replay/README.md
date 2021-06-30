# Replay Agent Actions and Inputs
This module shows how you can write your own experiments that will allow you to replay the actions of your agent after they have been previously run using the `ReplayAgent` wrapper which you can find in `zoo/policies/replay_agent.py` and its registry in `zoo/policies/__init__.py`. 

## Wrapping your Social Agent
You need to wrap your social agent using the `ReplayAgent` wrapper to save agent observations and actions at each step.
Checkout `examples/replay/replay_klws_agent.py` on an example on how you can write your own experiment to replay your agent:
```python
    from smarts.zoo.registry import make as zoo_make
    from zoo import policies
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set the parameters of replay-agent-v0 wrapper
    policies.replay_save_dir = save_dir
    policies.replay_read = not write

    # This is how you can wrap an agent in replay-agent-v0 wrapper to store and load its inputs and actions
    # and replay it
    agent_spec = zoo_make(
        "zoo.policies:replay-agent-v0",
        save_directory=save_dir,
        id="agent_007",
        wrapped_agent_locator="zoo.policies:keep-left-with-speed-agent-v0",
        wrapped_agent_params={"speed": speed},
    )
    # copy the scenarios to the replay directory to make sure its not changed while replaying the agent actions
    copy_scenarios.copy_scenarios(save_dir, scenarios)

```
You may also need to wrap the social agent used in the `scenario.py` file of the scenario on which you run your experiment.
Like for `scenarios/loop`, you can wrap the `open_agent` and `keep_lane_agent` agents used like this, 
```python
    open_agent_actor = t.SocialAgentActor(
    name="open-agent",
    agent_locator="zoo.policies:replay-agent-v0",
        policy_kwargs={
            "save_directory": "./replay",
            "id": "agent_oa",
            "wrapped_agent_locator": "open_agent:open_agent-v0",
        },
    )

    laner_actor = t.SocialAgentActor(
        name="keep-lane-agent",
        agent_locator="zoo.policies:replay-agent-v0",
        policy_kwargs={
            "save_directory": "./replay",
            "id": "agent_kla",
            "wrapped_agent_locator": "zoo.policies:keep-lane-agent-v0",
        },
    )
```
## Setup and Running:
### External dependencies:
Ubuntu 18.04
Python 3.7.5 or higher
Eclipse SUMO 1.8.0 or higher

```bash
# 1. construct your virtual environment
cd <project>
./install_deps.sh # skip if you have python installed
python3.7 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r examples/replay/replay_requirements.txt

# 2. Create a directory where your agent actions and inputs will be store
mkdir ./klws_replay

# 3. Run the replay agent using the --write argument and with required agent's parameters (Like the klws_agent which requires you to pass in the speed parameter) to store the actions and inputs of agents to CRASH_DIR directory:
CRASH_DIR=./klws_replay
python3.7 examples/replay/replay_klws_agent.py scenarios/loop --save-dir $CRASH_DIR --speed 20 --write --headless

# 4. Now you can replay the agent's previous action by not using the --write to load the observations saved by the wrapper in CRASH_DIR directory:
python3.7 examples/replay/replay_klws_agent.py scenarios/loop --save-dir $CRASH_DIR --speed 20 --headless

```
## Generating a new crash using the ReplayAgent wrapper:
You can also use the `ReplayAgent` wrapper to find out where the crash of an agent is occurring by writing your own `crash_test.sh` script like the one in `bin/crash_test/crash_test.sh` (which for example runs multiple instances of `klws_agent` for different configurations of arguments to find where the crash is occurring) and use the `bin/crash/parallel_crash_test.sh` to run multiple `crash_test.sh` scripts in parallel.
This will allow you to find the source of your crash much faster and let you replay the episode where the crash occurred.

```bash
CRASH_DIR=./crash_test
# Run the parallel_crash_test.sh script like this:
# parallel_crash_trial.sh <save_dir> <parallel_runs> <total_trials> <max_steps_per_trial>
bash parallel_crash_trial.sh $CRASH_DIR 2 300 1500
# Load the agent argument which caused the crash. (See bin/crash_test/crash_test.sh on how speed.txt was created).
SPEED=$(cat "${CRASH_DIR}/speed.txt")
# Replay the generated crash
python examples/replay/replay_klws_agent.py scenarios/loop --save-dir $CRASH_DIR --speed $SPEED --headless
```
