"""This example shows the differences between each of the environments.

For the unformatted observation please see https://smarts.readthedocs.io/en/latest/sim/obs_action_reward.html.
"""
import warnings
from pathlib import Path

from tools.argument_parser import empty_parser

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.string import truncate
from smarts.env.configs.hiway_env_configs import EnvReturnMode
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from smarts.env.utils.action_conversion import ActionOptions
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.scenario_construction import build_scenarios

warnings.filterwarnings("ignore", category=UserWarning)

AGENT_ID = "agent"


def detail_environment(env: HiWayEnvV1, name: str):
    obs, _ = env.reset()

    print(f"-------- Format '{name}' ---------")
    print(f"Environment action space {env.action_space}")
    print(
        f"Environment observation space {truncate(str(env.observation_space), length=80)}",
    )
    print(f"Environment observation type {type(obs)}")
    print(f"Agent observation type {type(obs[AGENT_ID])}")
    observations, rewards, terminations, truncations, infos = env.step(
        {
            AGENT_ID: None
            if env.action_space is None
            else env.action_space.sample()[AGENT_ID]
        }
    )
    print(f"Environment infos type {type(rewards)}")
    print()


def main(*_, **kwargs):
    defaults = dict(
        agent_interfaces={AGENT_ID: AgentInterface.from_type(AgentType.Standard)},
        scenarios=[
            str(
                Path(__file__).absolute().parents[1]
                / "scenarios"
                / "sumo"
                / "figure_eight"
            )
        ],
        headless=True,
    )

    build_scenarios(defaults["scenarios"])

    # AKA: `gym.make("smarts.env:hiway-v1")`
    with HiWayEnvV1(
        # observation_options=ObservationOptions.multi_agent,
        # action_options=ActionOptions.multi_agent,
        **defaults,
    ) as env:
        detail_environment(env, "multi_agent")

    with HiWayEnvV1(
        observation_options=ObservationOptions.full,
        action_options=ActionOptions.full,
        **defaults,
    ) as env:
        detail_environment(env, "full")

    with HiWayEnvV1(
        observation_options=ObservationOptions.unformatted,
        action_options=ActionOptions.unformatted,
        **defaults,
    ) as env:
        detail_environment(env, "unformatted")

    with HiWayEnvV1(
        observation_options="unformatted",
        action_options="unformatted",
        environment_return_mode=EnvReturnMode.environment,
        **defaults,
    ) as env:
        detail_environment(env, "env return")


if __name__ == "__main__":
    parser = empty_parser("environment config")
    args = parser.parse_args()

    main()
