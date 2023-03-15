$ cd <path>/SMARTS/examples/rl/baseline_single
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -e ./../../../.[camera_obs,dev,doc]
$ pip install -e ./inference/
$ python3.8 train/run.py --mode=train --head




Do we know why the waypoint path appear and then disappear here?

Steps to reproduce:
$ git checkout zoo-7
$ cd <path>/SMARTS/examples/rl/baseline
$ python3.8 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip wheel
$ pip install -e ./../../../.[camera_obs,dev,doc]
$ pip install -e ./inference/
$ python3.8 train/run.py

Here we use the scenario: SMARTS/scenarios/sumo/platoon/merge_exit_agents_1

This example enables sumo-gui by default for debugging. Simulation runs until the lead vehicle (in blue colour) reaches near the exit ramp. Then, a top-down rgb image with the waypoints superimposed in green colour is plotted at each time step. Simply close each of the rgb image to proceed to the next step in the simulation.

The intended outcome is for the social vehicle to turn right into the exit ramp. The social vehicle uses action space ActionSpaceType.LaneWithContinuousSpeed. It always executes "change-lane-to-right" action when it is near the exit ramp (i.e., when the rgb images start to be plotted).

Stepping through the simulation, we see that the waypoint path leading towards the exit ramp appear (see first attached image), then disappear for one time step (see second attached image), before re-appearing again. I guess it reappears because the social vehicle continuously attempts to change lane to right. The question is why did the waypoint path leading towards the exit ramp disappear in one of the time step?

Try running a couple of episodes or rebuilding the scenario, if the issue is not seen in the first episode.