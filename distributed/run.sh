#!/bin/bash

# This script runs SMARTS actors and learner in distributed fashion.
# Each actor and learner runs in its own docker container in a separate terminal.

# Prior to running this script
# 1) Build the smarts docker image
#    $ cd /path/to/SMARTS/
#    $ docker build --network=host -t huaweinoah/smarts:distributed .
# 2) Install tmux
#    $ sudo apt-get install tmux

# Launch this script
# $ cd /path/to/SMARTS/
# $ ./distributed/run.sh

# To view the visualization, navigate to http://localhost:8081/ in your browser.


# User configurable variables
NUM_ACTORS=2
AGENT_IDS="Agent_001 Agent_002 Agent_003 Agent_004"
AGENT_POLICIES="keep_lane keep_lane keep_lane keep_lane"
SCENARIOS="/src/scenarios/intersections/6lane/"
EPISODES="50"

# Entrypoints
ENVISION="scl envision start -s ./scenarios -p 8081"
LEARNER="python3.7 /src/distributed/learner.py --agent_ids ${AGENT_IDS} --agent_policies ${AGENT_POLICIES}"
ACTOR="python3.7 /src/distributed/actor.py ${SCENARIOS} --episodes=${EPISODES} --agent_ids ${AGENT_IDS} --agent_policies ${AGENT_POLICIES}"

# Launch visualization
tmux new -d -s envision
COMMAND="docker run --rm --network=host --name=envision huaweinoah/smarts:distributed ${ENVISION}"
tmux send-keys -t "envision" "${COMMAND}" ENTER

# Launch learner
tmux new -d -s learner
COMMAND="docker run --rm --network=host --name=learner huaweinoah/smarts:distributed ${LEARNER}"
tmux send-keys -t "learner" "${COMMAND}" ENTER

# Launch actors
for ((id=0; id<$NUM_ACTORS; id++)); do
    tmux new -d -s "actor_${id}"
    COMMAND="docker run --rm --network=host --name=actor_${id} huaweinoah/smarts:distributed ${ACTOR}"
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done
