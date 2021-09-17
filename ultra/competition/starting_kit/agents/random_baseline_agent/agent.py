# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import ultra.adapters as adapters

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import (
    AgentInterface,
    DoneCriteria,
    NeighborhoodVehicles,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from typing import Dict


class RandomAgent(Agent):
    def act(self, observation):
        # Returns an action in the form of a NumPy array with shape (3,), where this
        # action represents the [throttle, brake, steering] of the agent.
        throttle = np.random.uniform(low=0.0, high=1.0)
        brake = np.random.uniform(low=0.0, high=1.0)
        steering = np.random.uniform(low=-1.0, high=1.0)
        return np.array([throttle, brake, steering])


def random_agent_info_adapter(
    observation: Observation, reward: float, info: Dict
) -> Dict:
    """Add whether the random agent reached the goal to its info."""
    info["random_agent_reached_goal"] = observation.events.reached_goal
    return info


def random_agent_reward_adapter(observation: Observation, reward: float) -> float:
    """Add the random agent's speed to its reward."""
    reward += observation.ego_vehicle_state.speed
    return reward


agent_spec = AgentSpec(
    interface=AgentInterface(
        debug=False,
        done_criteria=DoneCriteria(),
        max_episode_steps=200,
        neighborhood_vehicles=NeighborhoodVehicles(radius=200.0),
        waypoints=Waypoints(lookahead=20),
        action=ActionSpaceType.Continuous,
    ),
    agent_builder=RandomAgent,
    agent_params={},
    action_adapter=lambda action: action,
    info_adapter=random_agent_info_adapter,
    observation_adapter=adapters.default_observation_vector_adapter.adapt,
    reward_adapter=random_agent_reward_adapter,
)
