import subprocess
import sys
import os
import importlib.util
import shutil

from pathlib import Path, PurePath
from smarts.core.agent import Agent


class CompetitionAgent(Agent):
    def __init__(self, policy_path):
        req_file = os.path.join(policy_path, "requirements.txt")

        env_name = Path(policy_path).name  # name of the submission file
        root_path = Path(__file__).parents[2]  # Smarts main path

        self._policy_dir = policy_path
        self._comp_env_path = str(os.path.join(root_path, "competition_env"))
        self._sub_env_path = str(os.path.join(self._comp_env_path, env_name))

        Path.mkdir(Path(self._comp_env_path), exist_ok=True)

        if Path(self._sub_env_path).exists():
            shutil.rmtree(self._sub_env_path)
        Path.mkdir(Path(self._sub_env_path))

        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-t",
                    self._sub_env_path,
                    "-r",
                    req_file,
                ]
            )
            sys.path.append(self._sub_env_path)
        except:
            sys.exit("Failed to install requirement for Competition Agent")

        # insert submission path
        sys.path.insert(0, policy_path)

        # import policy module
        self._policy_path = str(os.path.join(policy_path, "policy.py"))
        policy_spec = importlib.util.spec_from_file_location(
            "competition_policy", self._policy_path
        )
        policy_module = importlib.util.module_from_spec(policy_spec)
        sys.modules["competition_policy"] = policy_module
        if policy_spec:
            policy_spec.loader.exec_module(policy_module)

        self._policy = policy_module.Policy()

        # delete competition policy module and remove path
        while policy_path in sys.path:
            sys.path.remove(policy_path)  # prevent duplication path remain in sys.path

        # remove all modules related to the policy_path
        for key, module in list(sys.modules.items()):
            if "__file__" in dir(module):
                module_path = module.__file__
                if module_path and (
                    self._policy_dir in module_path or self._sub_env_path in module_path
                ):
                    sys.modules.pop(key)
        del policy_module

    def act(self, obs):
        return self._policy.act(obs)

    def close_env(self, remove_all_env=False):
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
