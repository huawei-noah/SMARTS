import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
except ImportError as exc:
    raise ImportError("Please install smarts[examples] or -e .[examples].")


from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.configs.base_config import (
    EnvironmentArguments,
    EnvironmentConfiguration,
)
from smarts.env.configs.hiway_env_configs import HiWayEnvV1Config
from smarts.env.utils.action_conversion import ActionOptions
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.zoo import registry
from smarts.zoo.agent_spec import AgentSpec


@dataclass
class ExperimentEnvCfg(EnvironmentConfiguration):
    id: str = "smarts.env:hiway-v1"
    """The gymnasium environment name."""
    params: EnvironmentArguments = field(
        default_factory=lambda: HiWayEnvV1Config(
            scenarios=[
                str(
                    Path(__file__).absolute().parents[1] / "scenarios" / "sumo" / "loop"
                ),
                str(
                    Path(__file__).absolute().parents[1]
                    / "scenarios"
                    / "sumo"
                    / "figure_eight"
                ),
            ],
            action_options=ActionOptions.unformatted,
            agent_interfaces={},
        )
    )


@dataclass
class AgentCfg:
    locator: str = "__main__:keep_lane_control-v0"
    params: dict = field(default_factory=lambda: dict(max_episode_steps=1000))


@dataclass
class ExperimentCfg:
    agent_cfg: AgentCfg = field(default_factory=AgentCfg)
    env_cfg: EnvironmentConfiguration = field(default_factory=ExperimentEnvCfg)
    episodes: int = 5
    num_agents: int = 4
    show_config: bool = True


cs = ConfigStore.instance()
cs.store(name="experiment_cfg", node=ExperimentCfg)


class KeepLaneAgent(Agent):
    def act(self, obs):
        val = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
        return random.choice(val)


def kla_entrypoint(*, max_episode_steps: int) -> AgentSpec:
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=KeepLaneAgent,
    )


# This registers "__main__:keep_lane_control-v0"
registry.register("keep_lane_control-v0", kla_entrypoint)


@hydra.main(
    config_path=str(Path(__file__).parent.absolute() / "configs/hydra/control"),
    config_name="experiment_cfg",
    version_base=None,
)
def main(experiment_config: ExperimentCfg) -> None:
    typed_experiment_config: ExperimentCfg = OmegaConf.to_object(cfg=experiment_config)
    if typed_experiment_config.show_config:
        print()
        print("# Current configuration")
        print("# =====================")
        print(OmegaConf.to_yaml(cfg=experiment_config))

    AGENT_IDS = ["Agent %i" % i for i in range(typed_experiment_config.num_agents)]

    build_scenarios(scenarios=typed_experiment_config.env_cfg.params.scenarios)

    agent_specs = {
        agent_id: registry.make(
            locator=typed_experiment_config.agent_cfg.locator,
            **typed_experiment_config.agent_cfg.params,
        )
        for agent_id in AGENT_IDS
    }
    # This is the one point of pain that the agent interfaces are needed
    #  but the agent should be constructed by the `smarts.zoo` separately.
    env_params = asdict(typed_experiment_config.env_cfg.params)
    if "agent_interfaces" in env_params:
        # I would consider allowing agent interface to also be just a dictionary.
        env_params["agent_interfaces"] = {
            a_id: a_intrf.interface for a_id, a_intrf in agent_specs.items()
        }

    env = gym.make(
        id=typed_experiment_config.env_cfg.id,
        **env_params,
    )

    for episode in episodes(n=typed_experiment_config.episodes):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        observations, _ = env.reset()
        episode.record_scenario(env.scenario_log)

        terminateds = {"__all__": False}
        while not terminateds["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode.record_step(observations, rewards, terminateds, truncateds, infos)

    env.close()


if __name__ == "__main__":
    main()
