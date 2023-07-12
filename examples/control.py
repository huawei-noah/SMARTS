import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Final, List

import gymnasium as gym

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
except ImportError as exc:
    raise ImportError("Please install smarts[examples] or -e .[examples].")


from smarts.core.utils.episodes import episodes
from smarts.env.configs.base_config import (
    EnvironmentArguments,
    EnvironmentConfiguration,
)
from smarts.env.configs.hiway_env_configs import HiWayEnvV1Configuration
from smarts.env.utils.action_conversion import ActionOptions
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.zoo import registry

sys.path.insert(0, str(Path(__file__).parents[1].absolute()))
from examples.tools.primitive_agents import (
    cvpa_entrypoint,
    kla_entrypoint,
    open_entrypoint,
    standard_lane_follower_entrypoint,
    trajectory_tracking_entrypoint,
)


@dataclass
class ExperimentEnvCfg(EnvironmentConfiguration):
    id: str = "smarts.env:hiway-v1"
    """The gymnasium environment name."""
    params: EnvironmentArguments = field(
        default_factory=lambda: HiWayEnvV1Configuration(
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
    params: dict = field(default_factory=dict)
    repeat: int = 0


@dataclass
class ExperimentCfg:
    episodes: int
    """This indicates how many times the environment will reset."""
    show_config: bool
    """If true the yaml structure of the configuration for this run will be printed."""
    minimum_steps: int = 1
    """The minimum number of steps to run before reset. This can be used to run egoless."""
    agents_configs: Dict[str, AgentCfg] = field(default_factory=lambda: {})
    """The configuration of the agents to include in this experiment."""
    env_config: EnvironmentConfiguration = field(default_factory=ExperimentEnvCfg)
    """The environment configuration for the environment used in this experiment. Typically 'smarts.env:hiway-v1'."""


# This registers "__main__:keep_lane_control-v0"
registry.register("keep_lane_control-v0", kla_entrypoint)
registry.register(locator="open_agent-v0", entry_point=open_entrypoint)
registry.register(locator="chase_via_points-v0", entry_point=cvpa_entrypoint)
registry.register(
    locator="trajectory_tracking-v0", entry_point=trajectory_tracking_entrypoint
)
registry.register(
    locator="standard_lane_follower-v0", entry_point=standard_lane_follower_entrypoint
)

CONFIG_LOCATION: Final[str] = str(Path(__file__).parent.absolute() / "configs/control")
cs = ConfigStore.instance()
cs.store(name="base_experiment", node=ExperimentCfg, group=None)


@hydra.main(
    config_path=CONFIG_LOCATION,
    config_name="experiment_default",
    version_base=None,
)
def main(experiment_config: ExperimentCfg) -> None:
    typed_experiment_config: ExperimentCfg = OmegaConf.to_object(cfg=experiment_config)
    print(f"Loading configuration from `{CONFIG_LOCATION}`")
    if typed_experiment_config.show_config:
        print()
        print("# Current used configuration")
        print("# ==========================\n")
        print(OmegaConf.to_yaml(cfg=experiment_config))
        print("# ==========================")

    assert len(typed_experiment_config.agents_configs) == len(
        set(name for name, _ in typed_experiment_config.agents_configs.items())
    ), f"Agent names must be unique in configuration."

    build_scenarios(scenarios=typed_experiment_config.env_config.params.scenarios)

    agent_specs = {
        name: registry.make(
            locator=cfg.locator,
            **cfg.params,
        )
        for name, cfg in typed_experiment_config.agents_configs.items()
    }
    # This is the one point of pain that the agent interfaces are needed
    #  but the agent should be constructed by the `smarts.zoo` separately.
    env_params = asdict(typed_experiment_config.env_config.params)
    if "agent_interfaces" in env_params:
        # I would consider allowing agent interface to also be just a dictionary.
        env_params["agent_interfaces"] = {
            a_id: a_intrf.interface for a_id, a_intrf in agent_specs.items()
        }

    env = gym.make(
        id=typed_experiment_config.env_config.id,
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
        steps = 0
        while (
            not terminateds["__all__"] or steps < typed_experiment_config.minimum_steps
        ):
            steps += 1
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode.record_step(observations, rewards, terminateds, truncateds, infos)

    env.close()


if __name__ == "__main__":
    main()
