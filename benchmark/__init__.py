import gym

from pathlib import Path
from smarts.core.scenario import Scenario
from smarts.core.agent import AgentSpec

from benchmark.agents import load_config


def gen_config(**kwargs):
    scenario_path = Path(kwargs["scenario"]).absolute()
    agent_missions_count = Scenario.discover_agent_missions_count(scenario_path)
    if agent_missions_count == 0:
        agent_ids = ["default_policy"]
    else:
        agent_ids = [f"AGENT-{i}" for i in range(agent_missions_count)]

    config = load_config(kwargs["config_file"], mode=kwargs.get("mode", "training"))
    agents = {agent_id: AgentSpec(**config["agent"]) for agent_id in agent_ids}

    config["env_config"].update(
        {
            "seed": 42,
            "scenarios": [str(scenario_path)],
            "headless": kwargs["headless"],
            "agent_specs": agents,
        }
    )

    obs_space, act_space = config["policy"][1:3]
    tune_config = config["run"]["config"]

    if kwargs["paradigm"] == "centralized":
        config["env_config"].update(
            {
                "obs_space": gym.spaces.Tuple([obs_space] * agent_missions_count),
                "act_space": gym.spaces.Tuple([act_space] * agent_missions_count),
                "groups": {"group": agent_ids},
            }
        )
        tune_config.update(config["policy"][-1])
    else:
        policies = {}
        for k in agents:
            policies[k] = config["policy"][:-1] + (
                {**config["policy"][-1], "agent_id": k},
            )
        tune_config.update(
            {
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": lambda agent_id: agent_id,
                }
            }
        )

    return config
