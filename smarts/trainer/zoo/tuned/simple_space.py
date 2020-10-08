import gym
import numpy as np

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec
from smarts.env.custom_observations import lane_ttc_observation_adapter

# This action space should match the input to the action(..) function below.
ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
)

OBSERVATION_SPACE = lane_ttc_observation_adapter.space


# env::observation reshaping: for output
def observation_adapter(env_observation):
    return lane_ttc_observation_adapter.transform(env_observation)


# env::reward shaping: for output
def reward_adapter(env_obs, env_reward):
    """Actually, it is a reward shaping function"""

    return env_reward


# env::action wrapper: for input
def action_adapter(model_action):
    """For environment."""

    assert len(model_action) == 3
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])


def rllib_trainable_agent():
    return {
        "agent_spec": AgentSpec(
            interface=AgentInterface.from_type(AgentType.Standard),
            observation_adapter=observation_adapter,
            reward_adapter=reward_adapter,
            action_adapter=action_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }
