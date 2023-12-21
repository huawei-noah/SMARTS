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
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import EnvSpec

from envision import etypes as envision_types
from envision.client import Client as Envision
from envision.data_formatter import EnvisionDataFormatterArgs
from smarts.core import current_seed
from smarts.core import seed as smarts_seed
from smarts.core.agent_interface import AgentInterface
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.env.configs.hiway_env_configs import (
    EnvReturnMode,
    ScenarioOrder,
    SumoOptions,
)
from smarts.env.utils.action_conversion import ActionOptions, ActionSpacesFormatter
from smarts.env.utils.observation_conversion import (
    ObservationOptions,
    ObservationSpacesFormatter,
)

DEFAULT_VISUALIZATION_CLIENT_BUILDER = partial(
    Envision,
    endpoint=None,
    output_dir=None,
    headless=False,
    data_formatter_args=EnvisionDataFormatterArgs("base", enable_reduction=False),
)


class HiWayEnvV1(gym.Env):
    """A generic environment for various driving tasks simulated by SMARTS.

    Args:
        scenarios (Sequence[str]): A list of scenario directories that
            will be simulated.
        agent_interfaces (Dict[str, AgentInterface]): Specification of the agents
            needs that will be used to configure the environment.
        sim_name (str, optional): Simulation name. Defaults to None.
        scenarios_order (ScenarioOrder, optional): Configures the order of
            scenarios provided over successive resets. See :class:`~smarts.env.configs.hiway_env_configs.ScenarioOrder`.
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        visdom (bool): Deprecated. Use SMARTS_VISDOM_ENABLED.
        fixed_timestep_sec (float, optional): Step duration for
            all components of the simulation. May be None if time deltas
            are externally-driven. Defaults to None.
        seed (int, optional): Random number generator seed. Defaults to 42.
        sumo_options (SumoOptions, Dict[str, Any]): The configuration for the
            sumo instance. A dictionary with the fields can be used instead.
            See :class:`~smarts.env.configs.hiway_env_configs.SumoOptions`.
        visualization_client_builder: A method that must must construct an
            object that follows the Envision interface. Allows tapping into a
            direct data stream from the simulation.
        observation_options (ObservationOptions, str): Defines the options
            for how the formatting matches the observation space. String version
            can be used instead. See :class:`~smarts.env.utils.observation_conversion.ObservationOptions`. Defaults to
            :attr:`~smarts.env.utils.observation_conversion.ObservationOptions.default`.
        action_options (ActionOptions, str): Defines the options
            for how the formatting matches the action space. String version
            can be used instead. See :class:`~smarts.env.utils.action_conversion.ActionOptions`. Defaults to
            :attr:`~smarts.env.utils.action_conversion.ActionOptions.default`.
        environment_return_mode (EnvReturnMode, str): This configures between the environment
            step return information (i.e. reward means the environment reward) and the per-agent
            step return information (i.e. reward means rewards as key-value per agent). Defaults to
            :attr:`~smarts.env.configs.hiway_env_configs.EnvReturnMode.per_agent`.
    """

    metadata = {
        "render_modes": ["rgb_array"],
    }
    """Metadata for gym's use."""

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
        agent_interfaces: Dict[str, Union[Dict[str, Any], AgentInterface]],
        sim_name: Optional[str] = None,
        scenarios_order: ScenarioOrder = ScenarioOrder.default,
        headless: bool = False,
        visdom: bool = False,
        fixed_timestep_sec: float = 0.1,
        seed: int = 42,
        sumo_options: Union[Dict[str, Any], SumoOptions] = SumoOptions(),
        visualization_client_builder: partial = DEFAULT_VISUALIZATION_CLIENT_BUILDER,
        observation_options: Union[
            ObservationOptions, str
        ] = ObservationOptions.default,
        action_options: Union[ActionOptions, str] = ActionOptions.default,
        environment_return_mode: Union[EnvReturnMode, str] = EnvReturnMode.default,
        render_mode: Optional[str] = None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        smarts_seed(seed)
        self._agent_interfaces: Dict[str, AgentInterface] = {
            a_id: (
                a_interface
                if isinstance(a_interface, AgentInterface)
                else AgentInterface(**a_interface)
            )
            for a_id, a_interface in agent_interfaces.items()
        }
        self._dones_registered = 0

        scenarios = [str(Path(scenario).resolve()) for scenario in scenarios]
        self._scenarios_iterator = Scenario.scenario_variations(
            scenarios,
            list(agent_interfaces.keys()),
            shuffle_scenarios=scenarios_order == ScenarioOrder.scrambled,
        )

        visualization_client = None
        if not headless:
            visualization_client = visualization_client_builder(
                headless=headless,
                sim_name=sim_name,
            )
            preamble = envision_types.Preamble(scenarios=scenarios)
            visualization_client.send(preamble)

        self._env_renderer = None
        self.render_mode = render_mode

        traffic_sims = []
        if Scenario.any_support_sumo_traffic(scenarios):
            from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

            if is_dataclass(sumo_options):
                sumo_options = asdict(sumo_options)
            sumo_traffic = SumoTrafficSimulation(
                headless=sumo_options["headless"],
                time_resolution=fixed_timestep_sec,
                num_external_sumo_clients=sumo_options["num_external_clients"],
                sumo_port=sumo_options["port"],
                auto_start=sumo_options["auto_start"],
            )
            traffic_sims += [sumo_traffic]
        smarts_traffic = LocalTrafficProvider()
        traffic_sims += [smarts_traffic]

        if isinstance(environment_return_mode, str):
            self._environment_return_mode = EnvReturnMode[environment_return_mode]
        else:
            self._environment_return_mode = environment_return_mode

        if isinstance(action_options, str):
            action_options = ActionOptions[action_options]
        self._action_formatter = ActionSpacesFormatter(
            agent_interfaces, action_options=action_options
        )
        self.action_space = self._action_formatter.space

        if isinstance(observation_options, str):
            observation_options = ObservationOptions[observation_options]
        self._observations_formatter = ObservationSpacesFormatter(
            agent_interfaces, observation_options
        )
        self.observation_space = self._observations_formatter.space

        from smarts.core.smarts import SMARTS

        self._smarts = SMARTS(
            agent_interfaces=agent_interfaces,
            traffic_sims=traffic_sims,
            envision=visualization_client,
            visdom=visdom,
            fixed_timestep_sec=fixed_timestep_sec,
        )

    def step(
        self, action: ActType
    ) -> Union[
        Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]],
        Tuple[
            Dict[str, Any],
            Dict[str, float],
            Dict[str, bool],
            Dict[str, bool],
            Dict[str, Any],
        ],
    ]:
        """Run one time-step of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            (dict, SupportsFloat, bool, bool, dict):
                - observation. An element of the environment's :attr:`observation_space` as the
                    next observation due to the agent actions. This observation will change based on
                    the provided :attr:`agent_interfaces`. Check :attr:`observation_space` after
                    initialization.
                - reward. The reward as a result of taking the action.
                - terminated. Whether the agent reaches the terminal state (as defined under the MDP of the task)
                    which can be positive or negative. An example is reaching the goal state. If true, the user needs to call :meth:`reset`.
                - truncated. Whether the truncation condition outside the scope of the MDP is satisfied.
                    Typically, this is a time-limit, but could also be used to indicate an agent physically going out of bounds.
                    Can be used to end the episode prematurely before a terminal state is reached.
                    If true, the user needs to call :meth:`reset`.
                - info. Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                    This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                    hidden from observations, or individual reward terms that are combined to produce the total reward.
        """
        assert isinstance(action, dict) and all(
            isinstance(key, str) for key in action.keys()
        ), "Expected Dict[str, Any]"

        formatted_action = self._action_formatter.format(action)
        observations, rewards, dones, extras = self._smarts.step(formatted_action)

        info = {
            agent_id: {
                "score": agent_score,
                "env_obs": observations[agent_id],
                "done": dones[agent_id],
                "reward": rewards[agent_id],
                "map_source": self._smarts.scenario.road_map.source,
            }
            for agent_id, agent_score in extras["scores"].items()
        }

        if self._env_renderer is not None:
            self._env_renderer.step(observations, rewards, dones, info)

        for done in dones.values():
            self._dones_registered += 1 if done else 0

        dones["__all__"] = self._dones_registered >= len(self._agent_interfaces)

        assert all("score" in v for v in info.values())

        if self._environment_return_mode == EnvReturnMode.environment:
            return (
                self._observations_formatter.format(observations),
                sum(r for r in rewards.values()),
                dones["__all__"],
                dones["__all__"],
                info,
            )
        elif self._environment_return_mode == EnvReturnMode.per_agent:
            observations = self._observations_formatter.format(observations)
            if (
                self._observations_formatter.observation_options
                == ObservationOptions.full
            ):
                dones = {**{id_: False for id_ in observations}, **dones}
                return (
                    observations,
                    {**{id_: np.nan for id_ in observations}, **rewards},
                    dones,
                    dones,
                    info,
                )
            else:
                return (
                    observations,
                    rewards,
                    dones,
                    dones,
                    info,
                )
        raise RuntimeError(
            f"Invalid observation configuration using {self._environment_return_mode}"
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.
        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalized policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.
        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or `/dev/urandom`).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
            options (dict, optional): Additional information to specify how the environment is reset (optional,
                depending on the specific environment). Forwards to :meth:`~smarts.core.smarts.SMARTS.reset`.
                - "scenario" (:class:`~smarts.sstudio.sstypes.scenario.Scenario`): An explicit scenario to reset to. The default is a scenario from the scenario iter.
                - "start_time" (float): Forwards the start time of the current scenario. The default is 0.

        Returns:
            dict: observation. Observation of the initial state. This will be an element of :attr:`observation_space`
                 and is analogous to the observation returned by :meth:`step`.
            dict: info. This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed, options=options)
        options = options or {}
        scenario = options.get("scenario")
        if scenario is None:
            scenario = next(self._scenarios_iterator)

        self._dones_registered = 0
        observations = self._smarts.reset(
            scenario, start_time=options.get("start_time", 0)
        )
        info = {
            agent_id: {
                "score": 0,
                "env_obs": agent_obs,
                "done": False,
                "reward": 0,
                "map_source": self._smarts.scenario.road_map.source,
            }
            for agent_id, agent_obs in observations.items()
        }

        if self._env_renderer is not None:
            self._env_renderer.reset(observations)

        if seed is not None:
            smarts_seed(seed)
        return self._observations_formatter.format(observations), info

    def render(
        self,
    ) -> Optional[Union[gym.core.RenderFrame, List[gym.core.RenderFrame]]]:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.
        The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
        ways to implement the render modes. In addition, list versions for most render modes is achieved through
        `gymnasium.make` which automatically applies a wrapper to collect rendered frames.

        Note:
            As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
            should be initialized in ``__init__``.

        By convention, if the :attr:`render_mode` is:
            - None (default): no render is computed.
            - `human`: The environment is continuously rendered in the current display or terminal,
                usually for human consumption. This rendering should occur during :meth:`step` and
                :meth:`render` doesn't need to be called. Returns ``None``.
            - `rgb_array`: Return a single frame representing the current state of the environment.
                A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for
                an x-by-y pixel image.
            - `ansi`: Return a strings (``str``) or ``StringIO.StringIO`` containing a
                terminal-style text representation for each time step. The text can include
                newlines and ANSI escape sequences (e.g. for colors).
            - `rgb_array_list` and `ansi_list`: List based version of render modes are possible
                (except Human) through the wrapper, :py:class:`gymnasium.wrappers.RenderCollection`
                that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
                The frames collected are popped after :meth:`render` is called or :meth:`reset`.

        Note:
            Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.
        """
        if self.render_mode == "rgb_array":
            if self._env_renderer is None:
                from smarts.env.utils.record import AgentCameraRGBRender

                self._env_renderer = AgentCameraRGBRender(self)

            return self._env_renderer.render(env=self)

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
            gym.Env: The base non-wrapped :class:`gymnasium.Env` instance
        """
        return self

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal random number generator that if not set will initialize with a random seed.

        Returns:
            The internal instance of :class:`np.random.Generator`.
        """
        return super().np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec`.

        Returns:
            A string identifying the environment.
        """
        return super().__str__()

    def __enter__(self):
        """Support with-statement for the environment."""
        return super().__enter__()

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
            (Dict[str, Union[float,str]]): A dictionary with the following keys.
                `fixed_timestep_sec` - Simulation time-step.
                `scenario_map` - Name of the current scenario.
                `scenario_traffic` - Traffic spec(s) used.
                `mission_hash` - Hash identifier for the current scenario.
        """

        scenario = self._smarts.scenario
        return {
            "fixed_timestep_sec": self._smarts.fixed_timestep_sec,
            "scenario_map": scenario.name,
            "scenario_traffic": ",".join(map(os.path.basename, scenario.traffic_specs)),
            "mission_hash": str(hash(frozenset(scenario.missions.items()))),
        }

    @property
    def smarts(self):
        """Gives access to the underlying simulator. Use this carefully.

        Returns:
            smarts.core.smarts.SMARTS: The smarts simulator instance.
        """
        return self._smarts

    @property
    def seed(self):
        """Returns the environment seed.

        Returns:
            int: Environment seed.
        """
        return current_seed()
