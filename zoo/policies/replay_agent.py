"""
This file contains an agent used for replaying an agent.
"""
import os
import pickle
from pathlib import Path

from smarts.core.agent import Agent, AgentSpec

agent_index = 0


class ReplayAgent(Agent):
    """A helper agent that wraps another agent to allow replay of the agent actions"""

    def __init__(self, save_directory, id, read: bool, internal_spec: AgentSpec):
        import smarts.core

        if smarts.core.current_seed() == None:
            smarts.core.seed(42)

        self.save_directory = save_directory
        self._base_agent = internal_spec.build_agent()
        self._file = None
        self._read = False

        global agent_index
        self.id = f"{id}_{agent_index}"
        agent_index += 1

        abs_path = os.path.abspath(save_directory)
        self._read = read
        # if "r" in file_mode:
        #     self._read = True
        file_mode = "wb" if not read else "rb"
        path = Path(f"{abs_path}/{self.id}")
        os.makedirs(abs_path, exist_ok=True)
        self._file = path.open(mode=file_mode)

    def __del__(self):
        if self._file:
            self._file.close()

    def act(self, obs):
        action = None
        if self._read:
            base_action = self._base_agent.act(obs)
            try:
                action = pickle.load(self._file)
                assert action == base_action
            except AssertionError as e:
                raise e
            except Exception:
                action = base_action

        else:
            action = self._base_agent.act(obs)
            pickle.dump(action, self._file, protocol=1)

        return action
