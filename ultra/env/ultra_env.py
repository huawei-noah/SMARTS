from itertools import cycle
import gym, glob, yaml, subprocess
from smarts.core.scenario import Scenario
import socket, errno
from smarts.env.hiway_env import HiWayEnv
from ultra.baselines.adapter import BaselineAdapter


class UltraEnv(HiWayEnv):
    def __init__(
        self,
        agent_specs,
        scenario_info,
        headless,
        timestep_sec,
        seed,
        eval_mode=False,
        ordered_scenarios=False,
    ):
        self.timestep_sec = timestep_sec
        self.headless = headless
        self.scenario_info = scenario_info
        self.scenarios = self.get_task(scenario_info[0], scenario_info[1])
        if not eval_mode:
            _scenarios = glob.glob(f"{self.scenarios['train']}")
        else:
            _scenarios = glob.glob(f"{self.scenarios['test']}")

        self.ultra_scores = BaselineAdapter()
        super().__init__(
            scenarios=_scenarios,
            agent_specs=agent_specs,
            headless=headless,
            timestep_sec=timestep_sec,
            seed=seed,
            visdom=False,
        )

        if ordered_scenarios:
            scenario_roots = []
            for root in _scenarios:
                if Scenario.is_valid_scenario(root):
                    # The case that this is a scenario root
                    scenario_roots.append(root)
                else:
                    # The case that there this is a directory of scenarios: find each of the roots
                    scenario_roots.extend(Scenario.discover_scenarios(root))
            # Also see `smarts.env.HiwayEnv`
            self._scenarios_iterator = cycle(
                Scenario.variations_for_all_scenario_roots(
                    scenario_roots, list(agent_specs.keys())
                )
            )

    def assign_env_score(self, observation, highwayenv_score):
        ultra_score = self.ultra_scores.reward_adapter(observation, highwayenv_score)
        return ultra_score

    def get_task(self, task_id, task_level):
        with open("ultra/config.yaml", "r") as task_file:
            scenarios = yaml.safe_load(task_file)["tasks"]
            task = scenarios[f"task{task_id}"][task_level]
        return task

    @property
    def info(self):
        return {
            "scenario_info": self.scenario_info,
            "timestep_sec": self.timestep_sec,
            "headless": self.headless,
        }
