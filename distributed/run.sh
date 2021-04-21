#!/bin/bash

# This script runs SMARTS actors and learner in distributed fashion.
# Each actor and learner runs in its own docker container in a separate terminal.

# Prior to running this script
# 1) Build the smarts docker image
#    $ docker build --network=host -t huaweinoah/smarts:distributed .
# 2) Install tmux
#    $ sudo apt-get install tmux

# Launch this script
# 1) Run in a terminal
#    $ cd /path/to/SMARTS
#    $ ./distributed/run.sh
# 2) Attach to a tmux terminal
#    $ tmux attach -t learner
#    $ tmux attach -t actor_<id>


# User configurable variables
NUM_ACTORS=2
AGENT_IDS="Agent_001 Agent_002 Agent_003 Agent_004"
AGENT_POLICIES="keep_lane keep_lane keep_lane keep_lane"
SCENARIOS="/src/scenarios/intersections/6lane/"
EPISODES="50"

# Entrypoints
LEARNER="python3.7 /src/distributed/learner.py --agent_ids ${AGENT_IDS} --agent_policies ${AGENT_POLICIES}"
ACTOR="python3.7 /src/distributed/actor.py ${SCENARIOS} --headless --episodes=${EPISODES} --agent_ids ${AGENT_IDS} --agent_policies ${AGENT_POLICIES}"

# Launch learner
tmux new -d -s learner
COMMAND="docker run --rm --network=host huaweinoah/smarts:distributed ${LEARNER}"
tmux send-keys -t 'learner' "${COMMAND}" ENTER

# Launch actors
for ((id=0; id<$NUM_ACTORS; id++)); do
    tmux new -d -s "actor_${id}"
    COMMAND="docker run --rm --network=host huaweinoah/smarts:distributed ${ACTOR}"
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done
