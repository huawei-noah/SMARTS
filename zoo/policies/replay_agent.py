"""
This file contains an agent used for replaying an agent.
"""
import logging
import os
import pickle
from pathlib import Path

from smarts.core.agent import Agent, AgentSpec

agent_index = 0


class ReplayAgent(Agent):
    """A helper agent that wraps another agent to allow replay of the agent inputs and actions
    Checkout examples/replay/README.md on how to use it"""

    def __init__(self, save_directory, id, read: bool, internal_spec: AgentSpec):
        import smarts.core

        if smarts.core.current_seed() is None:
            smarts.core.seed(42)

        self.save_directory = save_directory
        self._base_agent = internal_spec.build_agent()
        self._logger = logging.getLogger(self.__class__.__name__)
        global agent_index
        self.id = f"{id}_{agent_index}"
        agent_index += 1

        abs_path = os.path.abspath(save_directory)
        self._read = read
        file_mode = "wb" if not read else "rb"
        path = Path(f"{abs_path}/{self.id}")
        os.makedirs(abs_path, exist_ok=True)
        try:
            self._file = path.open(mode=file_mode)
        except FileNotFoundError as e:
            assert self._read
            self._logger.error(
                f"The file which you are trying to be read does not exist. "
                f"Make sure the {save_directory} directory passed is correct and has the agent file which is being read"
            )
            raise e

    def __del__(self):
        if self._file:
            self._file.close()

    def act(self, obs):
        if self._read:
            base_action = self._base_agent.act(obs)
            try:
                action = pickle.load(self._file)
                assert action == base_action
            except AssertionError as e:
                self._logger.debug("The Base Agent's action and new action don't match")
                raise e
            except Exception as e:
                self._logger.error(
                    "Comparing the new action with the base agent action raise an unknown error"
                )
                print(e)
                action = base_action

        else:
            action = self._base_agent.act(obs)
            pickle.dump(action, self._file, protocol=1)

        return action
