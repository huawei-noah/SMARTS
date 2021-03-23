import importlib.resources as pkg_resources

import cross_rl_agent

from smarts.core.agent import AgentSpec
from smarts.zoo.registry import register

from .agent import RLAgent
from .cross_space import (
    action_adapter,
    cross_interface,
    get_aux_info,
    observation_adapter,
    reward_adapter,
)


def entrypoint():
    with pkg_resources.path(cross_rl_agent, "models") as model_path:
        return AgentSpec(
            interface=cross_interface,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
            agent_builder=lambda: RLAgent(
                load_path=str(model_path) + "/",
                policy_name="Soc_Mt_TD3Network",
            ),
        )


# RLAgent(load_path=model_path, policy_name="Soc_Mt_TD3Network")

register(locator="cross_rl_agent-v0", entry_point=entrypoint)
