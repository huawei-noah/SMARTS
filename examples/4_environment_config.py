import warnings
from pathlib import Path

import gymnasium as gym
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.configs.hiway_env_configs import EnvReturnMode
from smarts.env.utils.action_conversion import ActionOptions
from smarts.env.utils.observation_conversion import ObservationOptions
from tools.argument_parser import default_argument_parser

from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from smarts.core.utils.string import truncate

warnings.filterwarnings("ignore", category=UserWarning)

AGENT_ID = "agent"

def detail_environment(env: HiWayEnvV1, name: str):
    obs, infos = env.reset()

    print(f"-------- Format '{name}' ---------")
    print(f"Environment action space {env.action_space}")
    print(f"Environment observation space {truncate(str(env.observation_space), length=80)}", )
    print(f"Environment observation type {type(obs)}")
    print(f"Agent observation type {type(obs[AGENT_ID])}")
    obs, rewards, term, trunc, info = env.step({AGENT_ID: None if env.action_space is None else env.action_space.sample()[AGENT_ID]})
    print(f"Environment infos type {type(rewards)}")
    print()

def main():
    agent_interface = AgentInterface.from_type(AgentType.Standard)
    scenarios = [str(Path(__file__).absolute().parents[1] / "scenarios" / "sumo" / "loop")]

    with HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=True,
        # observation_options=ObservationOptions.multi_agent,
        # action_options=ActionOptions.multi_agent,
    ) as env:
        detail_environment(env, "multi_agent")

    with HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=True,
        observation_options=ObservationOptions.full,
        action_options=ActionOptions.full,
    ) as env:
        detail_environment(env, "full")

    with HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=True,
        observation_options=ObservationOptions.unformatted,
        action_options=ActionOptions.unformatted,
    ) as env:
        detail_environment(env, "unformatted")

    with HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=True,
        observation_options=ObservationOptions.unformatted,
        action_options=ActionOptions.unformatted,
        environment_return_mode=EnvReturnMode.environment,
    ) as env:
        detail_environment(env, "env return")




if __name__ == "__main__":
    parser = default_argument_parser("egoless")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[1] / "scenarios" / "sumo" / "loop")
        ]

    build_scenarios(scenarios=args.scenarios)

    main()
