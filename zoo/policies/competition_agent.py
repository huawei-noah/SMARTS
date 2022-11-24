import subprocess
import sys
import os
import importlib.util

from pathlib import Path, PurePath
from smarts.core.agent import Agent


class CompetitionAgent(Agent):
    def __init__(self, policy_path):
        req_file = os.path.join(policy_path, "requirements.txt")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", req_file]
            )
        except:
            print("Failed to install requirement for Competition Agent")

        # import policy.py module
        self._policy_path = str(os.path.join(policy_path, "policy.py"))
        policy_spec = importlib.util.spec_from_file_location(
            "competition_policy", self._policy_path
        )
        policy_module = importlib.util.module_from_spec(policy_spec)
        sys.modules["competition_policy"] = policy_module
        policy_spec.loader.exec_module(policy_module)

        self._policy = policy_module.Policy()

        # delete competition policy module
        sys.modules.pop("competition_policy")
        del policy_module

    def act(self, obs):
        return self._policy.act(obs)
