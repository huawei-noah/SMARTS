from typing import Dict

import numpy as np
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import (
    AgentInterface, DoneCriteria, NeighborhoodVehicles, Waypoints
)
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
import ultra.adapters as adapters


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
