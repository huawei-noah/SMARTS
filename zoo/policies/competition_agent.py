import os
import shutil

from pathlib import Path
from smarts.core.agent import Agent


class CompetitionAgent(Agent):
    def __init__(self, policy_path, policy):
        env_name = Path(policy_path).name  # name of the submission file
        root_path = Path(__file__).parents[2]  # Smarts main path

        self._policy_dir = policy_path
        self._comp_env_path = str(os.path.join(root_path, "competition_env"))
        self._sub_env_path = str(os.path.join(self._comp_env_path, env_name))

        self._policy = policy

    def act(self, obs):
        return self._policy.act(obs)

    def close(self, remove_all_env=False):
        shutil.rmtree(str(self._sub_env_path))
        if remove_all_env:
            shutil.rmtree(str(self._comp_env_path), ignore_errors=True)
