import os

from pathlib import Path
from smarts.core.agent import Agent


class CompetitionAgent(Agent):
    def __init__(self, policy_path, policy, at_exit):
        env_name = Path(policy_path).name  # name of the submission file
        root_path = Path(__file__).parents[2]  # Smarts main path

        self._policy_dir = policy_path
        self._comp_env_path = str(os.path.join(root_path, "competition_env"))
        self._sub_env_path = str(os.path.join(self._comp_env_path, env_name))

        self._policy = policy
        self._at_exit = at_exit

    def act(self, obs):
        return self._policy.act(obs)

    def close(self, remove_all_env=False):
        self._at_exit(
            self._policy_dir, self._comp_env_path, self._sub_env_path, remove_all_env
        )
