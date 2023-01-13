# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Set, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gym import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec

from envision.client import Client as Envision
from envision.data_formatter import EnvisionDataFormatterArgs
from smarts.core import seed as smarts_seed
from smarts.core.agent_interface import AgentInterface
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.visdom_client import VisdomClient
from smarts.env.wrappers.utils.observation_conversion import ObservationsSpaceFormat

DEFAULT_TIMESTEP = 0.1


class ScenarioOrder(Enum):
    Sequential = 0
    Scrambled = 1


@dataclass
class SumoOptions:
    num_external_clients: int = 0
    auto_start: bool = True
    headless: bool = True
    port: Optional[str] = None


DEFAULT_VISUALIZATION_CLIENT_BUILDER = partial(
    Envision,
    endpoint=None,
    output_dir=None,
    headless=False,
    data_formatter_args=EnvisionDataFormatterArgs("base", enable_reduction=False),
)


# TODO: Could not help the double layer joke here: highway-lowway huawei-laowei. Add a real name.
class HiWayEnvV1(gym.Env):
    """A generic environment for various driving tasks simulated by SMARTS."""

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""

    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None
    reward_range = (-float("inf"), float("inf"))
    spec: Optional[EnvSpec] = None

    # Set these in ALL subclasses
    action_space: spaces.Space
    observation_space: spaces.Space

    # Created
    _np_random: Optional[np.random.Generator] = None

    def __init__(
        self,
        scenarios: Sequence[str],
        agent_interfaces: Dict[str, AgentInterface],
        sim_name: Optional[str] = None,
        scenarios_order: bool = True,
        visdom: bool = False,
        headless: bool = False,
        fixed_timestep_sec: Optional[float] = None,
        sumo_options: SumoOptions = SumoOptions(),
        visualization_client_builder: partial = DEFAULT_VISUALIZATION_CLIENT_BUILDER,
        zoo_addrs: Optional[str] = None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._agent_interfaces = agent_interfaces
        self._dones_registered = 0
        if not fixed_timestep_sec:
            self._log.warning(
                "Fixed timestep not specified. Default set to `%ss`", DEFAULT_TIMESTEP
            )
            fixed_timestep_sec = DEFAULT_TIMESTEP

        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            list(agent_interfaces.keys()),
            shuffle_scenarios=scenarios_order == ScenarioOrder.Scrambled,
        )

        visualization_client = None
        if not headless:
            visualization_client = visualization_client_builder(
                headless=headless,
                sim_name=sim_name,
            )

        self._env_renderer = None

        visdom_client = None
        if visdom:
            visdom_client = VisdomClient()

        traffic_sims = []
        if Scenario.any_support_sumo_traffic(scenarios):
            sumo_traffic = SumoTrafficSimulation(
                headless=sumo_options.headless,
                time_resolution=fixed_timestep_sec,
                num_external_sumo_clients=sumo_options.num_external_clients,
                sumo_port=sumo_options.port,
                auto_start=sumo_options.auto_start,
            )
            traffic_sims += [sumo_traffic]
        smarts_traffic = LocalTrafficProvider()
        traffic_sims += [smarts_traffic]

        # TODO: set action space

        # TODO: set observation space
        self.observations_formatter = ObservationsSpaceFormat(agent_interfaces)
        self.observation_space = self.observations_formatter.space

        from smarts.core.smarts import SMARTS

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sims=traffic_sims,
            envision=visualization_client,
            visdom=visdom_client,
            fixed_timestep_sec=fixed_timestep_sec,
            zoo_addrs=zoo_addrs,
        )

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.
        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.
        .. versionchanged:: 0.26
            The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
            to users when the environment had terminated or truncated which is critical for reinforcement learning
            bootstrapping algorithms.
        Args:
            action (ActType): an action provided by the agent to update the environment state.
        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        assert isinstance(action, dict) and all(
            isinstance(key, str) for key in action.keys()
        ), "Expected Dict[str, any]"

        observations, rewards, dones, extras = self._smarts.step(action)

        infos = {
            agent_id: {
                "score": value,
                "env_obs": observations[agent_id],
                "done": dones[agent_id],
            }
            for agent_id, value in extras["scores"].items()
        }

        if self._env_renderer is not None:
            self._env_renderer.step(observations, rewards, dones, infos)

        for done in dones.values():
            self._dones_registered += 1 if done else 0

        dones["__all__"] = self._dones_registered >= len(self._agent_interfaces)

        assert all("score" in v for v in infos.values())
        return (
            self.observations_formatter.format(observations),
            rewards,
            dones["__all__"],
            dones["__all__"],
            infos,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.
        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.
        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.
        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.
        .. versionchanged:: v0.25
            The ``return_info`` parameter was removed and now info is expected to be returned.
        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)
        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed, options=options)
        scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        observations = self._smarts.reset(scenario)
        info = {}

        if self._env_renderer is not None:
            self._env_renderer.reset(observations)

        if seed is not None:
            smarts_seed(seed)
        return self.observations_formatter.format(observations), info

    def render(
        self,
    ) -> Optional[Union[gym.core.RenderFrame, List[gym.core.RenderFrame]]]:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.
        The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
        ways to implement the render modes. In addition, list versions for most render modes is achieved through
        `gymnasium.make` which automatically applies a wrapper to collect rendered frames.
        Note:
            As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
            should be initialised in ``__init__``.
        By convention, if the :attr:`render_mode` is:
        - None (default): no render is computed.
        - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
          This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called. Returns ``None``.
        - "rgb_array": Return a single frame representing the current state of the environment.
          A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
        - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
          for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
        - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
          wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
          The frames collected are popped after :meth:`render` is called or :meth:`reset`.
        Note:
            Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.
        .. versionchanged:: 0.25.0
            The render function was changed to no longer accept parameters, rather these parameters should be specified
            in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
        """
        raise NotImplementedError

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.
        This is critical for closing rendering windows, database or HTTP connections.
        """
        if self._smarts is not None:
            self._smarts.destroy()

    @property
    def unwrapped(self) -> gym.Env[ObsType, ActType]:
        """Returns the base non-wrapped environment.
        Returns:
            Env: The base non-wrapped :class:`gymnasium.Env` instance
        """
        return self

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.
        Returns:
            Instances of `np.random.Generator`
        """
        return super().np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.
        Returns:
            A string identifying the environment
        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False

    @property
    def agent_ids(self) -> Set[str]:
        """Agent ids of all agents that potentially will be in the environment.
        Returns:
            (Set[str]): Agent ids.
        """
        return set(self._agent_interfaces)

    @property
    def agent_interfaces(self) -> Dict[str, AgentInterface]:
        """Agent interfaces used for the environment.
        Returns:
            (Dict[str, AgentInterface]):
                Agent interface defining the agents affect on the observation and action spaces
                 of this environment.
        """
        return self._agent_interfaces

    @property
    def scenario_log(self) -> Dict[str, Union[float, str]]:
        """Simulation steps log.
        Returns:
            Dict[str, Union[float,str]]: A dictionary with the following keys.
                fixed_timestep_sec - Simulation timestep.
                scenario_map - Name of the current scenario.
                scenario_traffic - Traffic spec(s) used.
                mission_hash - Hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_traffic": ",".join(map(os.path.basename, scenario.traffic_specs)),
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    @property
    def scenario(self) -> Scenario:
        """Returns underlying scenario.
        Returns:
            Scenario: Current simulated scenario.
        """
        return self._smarts.scenario
