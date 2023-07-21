"""This is an example to show how SMARTS multi-agent works. This example uses the same kind of
agent multiple times but different agents with different action and observation shapes can be mixed
in."""
import random
import sys
from pathlib import Path

# This may be necessary to get the repository root into path
SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))

from examples.tools.argument_parser import empty_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo import registry
from smarts.zoo.agent_spec import AgentSpec


class RandomLaneAgent(Agent):
    def act(self, obs, **kwargs):
        return random.randint(0, 3)


def rla_entrypoint(max_episode_steps=1000):
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=RandomLaneAgent,
    )


def main():
    name = "random_lane_control-v0"
    print(f"=== Before registering `{name}` ===")
    print(registry.agent_registry)
    registry.register(
        name, rla_entrypoint
    )  # This registers "__main__:random_lane_control-v0"
    print(f"=== After registering `{name}` ===")
    print(registry.agent_registry)

    agent_spec = registry.make(locator=f"__main__:{name}")
    agent_interface = agent_spec.interface
    agent = agent_spec.build_agent()
    # alternatively this will build the agent
    agent, _ = registry.make_agent(
        locator=f"__main__:{name}" 
    )
    # just "random_lane_control-v0" also works because the agent has already been registered in this file.
    agent, _ = registry.make_agent(
        locator=name 
    )

    locator = "zoo.policies:chase-via-points-agent-v0"
    # Here is an example of using the module component of the locator to dynamically load agents:
    agent, _ = registry.make_agent(
        locator=locator
    )
    print(f"=== After loading `{locator}` ===")
    print(registry.agent_registry)

    
    ## This agent requires installation
    # agent, _ = registry.make_agent(
    #     locator="zoo.policies:discrete-soft-actor-critic-agent-v0"
    # )

    locator = "non_existing.module:md-v44"
    try:
        agent, _ = registry.make_agent(
            locator="non_existing.module:md-v44"
        )
    except (ModuleNotFoundError, ImportError):
        print(f"Such as with '{locator}'. Module resolution can fail if the module cannot be found "
              "from the PYTHONPATH environment variable apparent as `sys.path` in python.")



if __name__ == "__main__":
    parser = empty_parser(Path(__file__).stem)
    args = parser.parse_args()

    main()
