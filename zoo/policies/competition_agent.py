import subprocess
import sys
import os
import logging
import importlib.util
import shutil

from pathlib import Path, PurePath
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
        while self._sub_env_path in sys.path:
            sys.path.remove(self._sub_env_path)
        while self._policy_dir in sys.path:
            sys.path.remove(self._policy_dir)
        for key, module in list(sys.modules.items()):
            if "__file__" in dir(module):
                module_path = module.__file__
                if module_path and (
                    self._policy_dir in module_path or self._sub_env_path in module_path
                ):
                    sys.modules.pop(key)
        if remove_all_env:
            shutil.rmtree(self._comp_env_path, ignore_errors=True)
