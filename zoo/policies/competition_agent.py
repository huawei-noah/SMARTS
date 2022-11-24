from distutils.command.config import config
import subprocess
import sys
import os

from pathlib import Path
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

        sys.path.insert(0, policy_path)
        from policy import Policy

        self._policy = Policy()
        sys.path.remove(policy_path)

    def act(self, obs):
        return self._policy.act(obs)