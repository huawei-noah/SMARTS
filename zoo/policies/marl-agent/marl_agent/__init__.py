from pathlib import Path
import importlib.resources as pkg_resources

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.zoo.registry import register

from benchmark.agents import load_config
from benchmark.default_model import RLLibTFCheckpointPolicy, BASE_DIR

from . import checkpoints


VERSION = 0.1


def entrypoint(config_file=None, checkpoint=None, test_agent_id=None, algorithm=None):
    assert config_file is not None
    assert checkpoint is not None
    assert test_agent_id in [f"AGENT-{i}" for i in range(4)]

    with pkg_resources.path(checkpoints, checkpoint) as checkpoint_path:
        config = load_config(config_file)

        agent_spec = AgentSpec(
            **config["agent"],
            observation_adapter=config["env_config"]["observation_adapter"],
            reward_adapter=config["env_config"]["reward_adapter"],
            interface=AgentInterface(**config["interface"]),
            policy_builder=lambda: RLLibTFCheckpointPolicy(
                load_path=checkpoint_path,
                algorithm=algorithm,
                agent_id=test_agent_id,
                yaml_path=config_file,
            ),
        )

        return agent_spec


register(locator="marl-agent-v0", entry_point=entrypoint)
