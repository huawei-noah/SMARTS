# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from __future__ import annotations

import importlib.resources as pkg_resources
import logging
import os
import re
import warnings
from functools import cached_property
from typing import (
    Any,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from envision import etypes as envision_types
from envision.client import Client as EnvisionClient
from smarts import VERSION, assets
from smarts.core.actor_capture_manager import ActorCaptureManager
from smarts.core.id_actor_capture_manager import IdActorCaptureManager
from smarts.core.plan import Plan
from smarts.core.renderer_base import RendererBase
from smarts.core.sensors import SensorState
from smarts.core.simulation_local_constants import SimulationLocalConstants
from smarts.core.utils.core_logging import timeit
from smarts.core.utils.type_operations import TypeSuite

from . import config
from .actor import ActorRole, ActorState
from .agent_interface import AgentInterface
from .agent_manager import AgentManager
from .agents_provider import (
    AgentPhysicsProvider,
    AgentsProvider,
    DirectControlProvider,
    MotionPlannerProvider,
    TrajectoryInterpolationProvider,
)
from .bubble_manager import BubbleManager
from .controllers import ActionSpaceType
from .coordinates import BoundingBox, Point
from .external_provider import ExternalProvider
from .observations import Observation
from .provider import (
    ActorProviderTransition,
    Provider,
    ProviderManager,
    ProviderRecoveryFlags,
    ProviderState,
)
from .road_map import RoadMap
from .scenario import NavigationMission, Scenario
from .sensor_manager import SensorManager
from .signal_provider import SignalProvider
from .signals import SignalLightState, SignalState
from .simulation_frame import SimulationFrame
from .traffic_history_provider import TrafficHistoryProvider
from .traffic_provider import TrafficProvider
from .trap_manager import TrapManager
from .utils import pybullet
from .utils.core_math import rounder_for_dt
from .utils.custom_exceptions import RendererException
from .utils.id import Id
from .utils.pybullet import bullet_client as bc
from .vehicle import Vehicle
from .vehicle_index import VehicleIndex
from .vehicle_state import Collision, VehicleState, neighborhood_vehicles_around_vehicle

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s: {%(module)s} %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.ERROR,
)

MAX_PYBULLET_FREQ: Final[int] = 240


class SMARTSNotSetupError(Exception):
    """Represents a case where SMARTS cannot operate because it is not set up yet."""

    pass


class SMARTSDestroyedError(Exception):
    """Represents a case where SMARTS cannot operate because it is destroyed."""

    pass


class SMARTS(ProviderManager):
    """The core SMARTS simulator. This is the direct interface to all parts of the simulation.

    Args:
        agent_interfaces (Dict[str, AgentInterface]): The interfaces providing SMARTS with the understanding of what features each agent needs.
        traffic_sims (Optional[List[TrafficProvider]], optional): An optional list of traffic simulators for providing non-agent traffic. Defaults to None.
        envision (Optional[envision.client.Client], optional): An envision client for connecting to an envision visualization server. Defaults to None.
        visdom (Union[bool, Any], optional): Deprecated. Use SMARTS_VISDOM_ENABLED. A visdom client for connecting to a visdom visualization server.
        fixed_timestep_sec (Optional[float], optional): The fixed timestep that will be default if time is not otherwise specified at step. Defaults to 0.1.
        reset_agents_only (bool, optional): When specified the simulation will continue use of the current scenario. Defaults to False.
        external_provider (bool, optional): Creates a special provider `SMARTS.external_provider` that allows for inserting state. Defaults to False.
    """

    def __init__(
        self,
        agent_interfaces: Dict[str, AgentInterface],
        # traffic_sim is deprecated:  use traffic_sims instead
        traffic_sim: Optional[TrafficProvider] = None,
        traffic_sims: Optional[List[TrafficProvider]] = None,
        envision: Optional[EnvisionClient] = None,
        visdom: Optional[Union[bool, Any]] = False,
        fixed_timestep_sec: Optional[float] = 0.1,
        reset_agents_only: bool = False,
        external_provider: bool = False,
    ):
        conf = config()
        self._log = logging.getLogger(self.__class__.__name__)
        self._sim_id = Id.new("smarts")
        self._is_setup = False
        self._is_destroyed = False
        self._scenario: Optional[Scenario] = None
        self._renderer: RendererBase = None
        self._envision: Optional[EnvisionClient] = envision
        if visdom is not False and visdom is not None:
            warnings.warn(
                "Using visdom through the arguments is deprecated. Please use engine configuration or SMARTS_VISDOM_ENABLED."
            )
        else:
            visdom = None

        self._visdom: Any = None
        if conf("visdom", "enabled", cast=bool) or visdom is True:
            from smarts.visdom.visdom_client import VisdomClient

            self._visdom = VisdomClient(
                hostname=conf("visdom", "hostname"),
                port=conf("visdom", "port"),
            )
        elif not isinstance(self._visdom, bool):
            self._visdom = visdom
        self._external_provider: ExternalProvider = None
        self._resetting = False
        self._reset_required = False

        assert fixed_timestep_sec is None or fixed_timestep_sec > 0
        self.fixed_timestep_sec: Optional[float] = fixed_timestep_sec
        self._last_dt = fixed_timestep_sec

        self._elapsed_sim_time = 0.0
        self._total_sim_time = 0.0
        self._step_count = 0

        self._signal_provider = SignalProvider()

        self._agent_physics_provider = AgentPhysicsProvider(self)
        self._direct_control_provider = DirectControlProvider(self)
        self._motion_planner_provider = MotionPlannerProvider(self)
        self._traffic_history_provider = TrafficHistoryProvider()
        self._trajectory_interpolation_provider = TrajectoryInterpolationProvider(self)

        traffic_sims = traffic_sims or []
        traffic_sims.append(self._traffic_history_provider)
        if traffic_sim:
            warnings.warn(
                "SMARTS traffic_sim property has been deprecated in favor of traffic_sims.  Please update your code.",
                category=DeprecationWarning,
            )
            traffic_sims += [traffic_sim]

        self._provider_suite = TypeSuite(Provider)
        if external_provider:
            self._external_provider = ExternalProvider(self)
            self.add_provider(self._external_provider)
        self.add_provider(self._agent_physics_provider)
        self.add_provider(self._direct_control_provider)
        self.add_provider(self._motion_planner_provider)
        self.add_provider(self._trajectory_interpolation_provider)
        for traffic_sim in traffic_sims:
            recovery_flags = (
                ProviderRecoveryFlags.EPISODE_REQUIRED
                | ProviderRecoveryFlags.ATTEMPT_RECOVERY
                | ProviderRecoveryFlags.RELINQUISH_ACTORS
            )
            self.add_provider(traffic_sim, recovery_flags)
        self.add_provider(self._signal_provider)

        # We buffer provider state between steps to compensate for TRACI's timestep delay
        self._last_provider_state = None
        self._reset_agents_only = reset_agents_only  # a.k.a "teleportation"

        # For macOS GUI. See our `BulletClientMACOS` docstring for details.
        # from .utils.pybullet import BulletClientMACOS
        # self._bullet_client = BulletClientMACOS(pybullet.GUI)
        self._bullet_client = pybullet.SafeBulletClient(
            pybullet.DIRECT
        )  # pylint: disable=no-member

        # Set up indices
        self._sensor_manager = SensorManager()
        self._vehicle_index = VehicleIndex()
        self._agent_manager = AgentManager(self, agent_interfaces)

        # TODO: Should not be stored in SMARTS
        self._vehicle_collisions: Dict[str, List[Collision]] = dict()
        self._vehicle_states = []

        self._bubble_manager = None
        self._trap_manager: TrapManager = TrapManager()
        self._actor_capture_managers: List[ActorCaptureManager] = [
            self._trap_manager,
            IdActorCaptureManager(),
        ]

        self._ground_bullet_id = None
        self._map_bb = None

    def step(
        self,
        agent_actions: Dict[str, Any],
        time_delta_since_last_step: Optional[float] = None,
    ) -> Tuple[
        Dict[str, Observation],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, float]],
    ]:
        """Progress the simulation by a fixed or specified time.

        Args:
            agent_actions:
                Actions that the agents want to perform on their actors.
            time_delta_since_last_step:
                Overrides the simulation step length. Progress simulation time by the given amount.
                Note the time_delta_since_last_step param is in (nominal) seconds.
        Returns:
            The simulation step return as (observations, rewards, dones, infos).
        """
        if not self._is_setup:
            raise SMARTSNotSetupError("Must call reset() or setup() before stepping.")
        self._check_valid()
        assert not (
            self._fixed_timestep_sec and time_delta_since_last_step
        ), "cannot switch from fixed- to variable-time steps mid-simulation"

        try:
            with timeit("Last SMARTS Simulation Step", self._log.info):
                return self._step(agent_actions, time_delta_since_last_step)
        except (KeyboardInterrupt, SystemExit):
            # ensure we clean-up if the user exits the simulation
            self._log.info("Simulation was interrupted by the user.")
            self.destroy()
            raise  # re-raise the KeyboardInterrupt
        except RendererException:
            self.destroy()
            raise  # re-raise renderer not found exceptions without retrying.
        except Exception as e:
            self._log.error(
                "Simulation crashed with exception. Attempting to cleanly shutdown."
            )
            self._log.exception(e)
            if not config().get_setting("core", "debug", cast=bool):
                self.destroy()
            raise  # re-raise

    def _check_if_acting_on_active_agents(self, agent_actions):
        for agent_id in agent_actions.keys():
            if agent_id not in self._agent_manager.ego_agent_ids:
                self._log.warning(
                    f"Attempted to perform actions on non-existing agent, {agent_id} "
                )

    def _step(self, agent_actions, time_delta_since_last_step: Optional[float] = None):
        """Steps through the simulation while applying the given agent actions.
        Returns the observations, rewards, done, and infos signals.
        """

        # Due to a limitation of our traffic simulator(SUMO) interface(TRACI), we can
        # only observe traffic state of the previous simulation step.
        #
        # To compensate for this, we:
        #
        # 0. Advance the simulation clock
        # 1. Fetch social agent actions
        # 2. Step all providers and harmonize state
        # 3. Step bubble manager
        # 4. Calculate observation and reward
        # 5. Send observations to social agents
        # 6. Clear done agents
        # 7. Perform visualization
        #
        # In this way, observations and reward are computed with data that is
        # consistently with one step of latency and the agent will observe consistent
        # data.

        # 0. Advance the simulation clock.
        # It's been this long since our last step.
        self._last_dt = time_delta_since_last_step or self._fixed_timestep_sec or 0.1
        self._elapsed_sim_time = self._rounder(self._elapsed_sim_time + self._last_dt)

        # 1. Fetch agent actions
        with timeit("Fetching agent actions", self._log.debug):
            all_agent_actions = self._agent_manager.fetch_agent_actions(agent_actions)

        # 2. Step all providers and harmonize state
        with timeit("Stepping all providers and harmonizing state", self._log.debug):
            provider_state = self._step_providers(all_agent_actions)
        with timeit("Checking if all agents are active", self._log.debug):
            self._check_if_acting_on_active_agents(agent_actions)

        # 3. Step bubble manager and trap manager
        with timeit("Syncing vehicle index", self._log.debug):
            self._vehicle_index.sync()
        with timeit("Stepping through bubble manager", self._log.debug):
            self._bubble_manager.step(self)
        with timeit("Stepping through actor capture managers", self._log.debug):
            for actor_capture_manager in self._actor_capture_managers:
                actor_capture_manager.step(self)

        # 4. Calculate observation and reward
        # We pre-compute vehicle_states here because we *think* the users will
        # want these during their observation/reward computations.
        # This is a hack to give us some short term perf wins. Longer term we
        # need to expose better support for batched computations
        with timeit("Syncing provider state", self._log.debug):
            self._sync_smarts_and_provider_actor_states(provider_state)
            self._sensor_manager.clean_up_sensors_for_actors(
                set(v.actor_id for v in self._vehicle_states),
                renderer=self.renderer_ref,
            )

        # Reset frame state
        try:
            del self.cached_frame
        except AttributeError:
            pass
        self.cached_frame

        # Agents
        with timeit("Stepping through sensors", self._log.debug):
            self._sensor_manager.step(self.cached_frame, self.renderer_ref)

        if self.renderer_ref:
            # runs through the render pipeline (for camera-based sensors)
            # MUST perform this after _sensor_manager.step() above, and before observe() below,
            # so that all updates are ready before rendering happens per
            with timeit("Syncing the renderer", self._log.debug):
                self.renderer_ref.sync(self.cached_frame)

            # TODO OBSERVATIONS: this needs to happen between the base and rendered observations
            # with timeit("Running through the render pipeline", self._log.debug):
            #     self.renderer_ref.render()

        # TODO OBSERVATIONS: Need to split the observation generation
        with timeit("Calculating observations and rewards", self._log.debug):
            observations, rewards, scores, dones = self._agent_manager.observe()

        with timeit("Filtering response for ego", self._log.debug):
            response_for_ego = self._agent_manager.filter_response_for_ego(
                (observations, rewards, scores, dones)
            )

        # 5. Send observations to social agents
        with timeit("Sending observations to social agents", self._log.debug):
            self._agent_manager.send_observations_to_social_agents(observations)

        # 6. Clear done agents
        with timeit("Clearing done agents", self._log.debug):
            self._teardown_done_agents_and_vehicles(dones)

        # 7. Perform visualization
        with timeit("Trying to emit the envision state", self._log.debug):
            self._try_emit_envision_state(provider_state, observations, scores)
        with timeit("Trying to emit the visdom observations", self._log.debug):
            self._try_emit_visdom_obs(observations)

        observations, rewards, scores, dones = response_for_ego
        extras = dict(scores=scores)

        self._step_count += 1

        return observations, rewards, dones, extras

    def _teardown_done_agents_and_vehicles(self, dones):
        def done_vehicle_ids(dones):
            vehicle_ids = set()
            for agent_id, done in dones.items():
                if self._agent_manager.is_boid_agent(agent_id):
                    vehicle_ids.update(id_ for id_ in done if done[id_])
                elif done:
                    ids = self._vehicle_index.vehicle_ids_by_owner_id(agent_id)
                    # 0 if shadowing, 1 if active
                    assert len(ids) <= 1, f"{len(ids)} <= 1"
                    vehicle_ids.update(ids)

            return vehicle_ids

        def done_agent_ids(dones):
            agent_ids = set()
            for agent_id, done in dones.items():
                if self._agent_manager.is_boid_agent(agent_id):
                    if not self.agent_manager.is_boid_keep_alive_agent(
                        agent_id
                    ) and all(dones[agent_id].values()):
                        agent_ids.add(agent_id)
                elif done:
                    agent_ids.add(agent_id)

            return agent_ids

        # XXX: These can not be put inline because we do queries that must proceed
        #      the actual teardown.
        agents_to_teardown = done_agent_ids(dones)
        self._agent_manager.teardown_ego_agents(agents_to_teardown)
        self._agent_manager.teardown_social_agents(agents_to_teardown)

        vehicles_to_teardown = done_vehicle_ids(dones)
        self._teardown_vehicles(vehicles_to_teardown)

    def reset(
        self, scenario: Scenario, start_time: float = 0.0
    ) -> Dict[str, Observation]:
        """Reset the simulation, reinitialize with the specified scenario. Then progress the
         simulation up to the first time an agent returns an observation, or `start_time` if there
         are no agents in the simulation.

        Args:
            scenario (smarts.core.scenario.Scenario): The scenario to reset the simulation with.
            start_time (float):
                The initial amount of simulation time to skip. This has implications on all time
                dependent systems.

                .. note::

                    SMARTS simulates a step and then updates vehicle control.
                    If you want a vehicle to enter at exactly ``0.3`` with a step of ``0.1`` it means the
                    simulation should start at ``start_time==0.2``.

        Returns:
            Dict[str, Observation]. This observation is as follows...
                - If no agents: the initial simulation observation at `start_time`
                - If agents: the first step of the simulation with an agent observation
        """
        tries = config()("core", "reset_retries", cast=int) + 1
        first_exception = None
        for _ in range(tries):
            try:
                self._resetting = True
                return self._reset(scenario, start_time)
            except Exception as err:
                if not first_exception:
                    first_exception = err
            finally:
                self._resetting = False
        self._log.error("Failed to successfully reset after %i tries.", tries)
        raise first_exception

    def _reset(self, scenario: Scenario, start_time: float):
        self._check_valid()

        self._total_sim_time += self._elapsed_sim_time
        self._elapsed_sim_time = max(0, start_time)  # The past is not allowed
        self._step_count = 0

        if (
            scenario == self._scenario
            and self._reset_agents_only
            and not self._reset_required
        ):
            vehicle_ids_to_teardown = set()
            agent_ids = self._agent_manager.teardown_ego_agents()
            agent_ids |= self.agent_manager.teardown_social_agents()
            for agent_id in agent_ids:
                ids = self._vehicle_index.vehicle_ids_by_owner_id(agent_id)
                vehicle_ids_to_teardown |= set(ids)
            self._teardown_vehicles(set(vehicle_ids_to_teardown))
            self._reset_providers()
            for actor_capture_manager in self._actor_capture_managers:
                actor_capture_manager.reset(scenario, self)
            self._agent_manager.init_ego_agents()
            if self._renderer:
                self._renderer.reset()
        else:
            self.teardown()
            self._reset_providers()
            self.setup(scenario)

        if not self.local_constants.road_map.is_same_map(scenario.map_spec):
            del self.local_constants
        self.local_constants

        # Tell history provide to ignore vehicles if we have assigned mission to them
        self._traffic_history_provider.set_replaced_ids(
            m.vehicle_spec.veh_id
            for m in scenario.missions.values()
            if m and m.vehicle_spec
        )
        self._reset_required = False

        self._vehicle_states = [v.state for v in self._vehicle_index.vehicles]
        try:
            del self.cached_frame
        except AttributeError:
            pass
        observations, _, _, _ = self._agent_manager.observe()
        observations_for_ego = self._agent_manager.reset_agents(observations)

        # Visualization
        self._try_emit_visdom_obs(observations)

        while len(self._agent_manager.ego_agent_ids) and len(observations_for_ego) < 1:
            observations_for_ego, _, _, _ = self.step({})

        return observations_for_ego

    def setup(self, scenario: Scenario):
        """Setup the next scenario."""
        self._check_valid()
        self._scenario = scenario

        self._setup_bullet_client(self._bullet_client)
        provider_state = self._setup_providers(self._scenario)
        self._vehicle_index.load_vehicle_definitions_list(
            scenario.vehicle_definitions_filepath
        )

        self._agent_manager.setup_agents()
        self._bubble_manager = BubbleManager(scenario.bubbles, scenario.road_map)
        for actor_capture_manager in self._actor_capture_managers:
            actor_capture_manager.reset(scenario, self)
        if self._renderer or any(
            a.requires_rendering for a in self._agent_manager.agent_interfaces.values()
        ):
            self.renderer.setup(scenario)

        self._harmonize_providers(provider_state)
        self._last_provider_state = provider_state

        self._is_setup = True

    def add_provider(
        self,
        provider: Provider,
        recovery_flags: ProviderRecoveryFlags = ProviderRecoveryFlags.EXPERIMENT_REQUIRED,
    ):
        """Add a provider to the simulation. A provider is a co-simulator conformed to a common
        interface.
        """
        self._check_valid()
        assert isinstance(provider, Provider)
        provider.recovery_flags = recovery_flags
        self._provider_suite.insert(provider)
        if isinstance(provider, TrafficProvider):
            provider.set_manager(self)

    def remove_provider(self, requested_type_or_provider: Union[type, Provider]):
        """Remove a provider from the simulation.

        Args:
            requested_type (type | Provider): The type of the provider to remove or provider to remove.

        Returns:
            Optional[Provider]: The provider that was removed.
        """
        self._check_valid()
        out_provider = None
        if isinstance(requested_type_or_provider, type):
            out_provider = self._provider_suite.remove_by_type(
                requested_type_or_provider
            )
        elif isinstance(requested_type_or_provider, Provider):
            out_provider = self._provider_suite.remove(requested_type_or_provider)
        return out_provider

    def switch_ego_agents(self, agent_interfaces: Dict[str, AgentInterface]):
        """Change the ego agents in the simulation. Effective on the next reset."""
        self._check_valid()
        self._agent_manager.switch_initial_agents(agent_interfaces)
        self._is_setup = False

    def add_agent_with_mission(
        self, agent_id: str, agent_interface: AgentInterface, mission: NavigationMission
    ):
        """Add an agent to the simulation. The simulation will attempt to provide a vehicle for
        the agent.
        """
        self._check_valid()
        # TODO:  check that agent_id isn't already used...
        if self._trap_manager.add_trap_for_agent(
            agent_id, mission, self.road_map, self.elapsed_sim_time
        ):
            self._agent_manager.add_ego_agent(agent_id, agent_interface)
        else:
            self._log.warning(
                f"Unable to add entry trap for new agent '{agent_id}' with mission."
            )

    def add_agent_and_switch_control(
        self,
        vehicle_id: str,
        agent_id: str,
        agent_interface: AgentInterface,
        mission: NavigationMission,
    ) -> Vehicle:
        """Add the new specified ego agent and then take over control of the specified vehicle."""
        self._check_valid()
        self.agent_manager.add_ego_agent(agent_id, agent_interface, for_trap=False)
        vehicle = self.switch_control_to_agent(
            vehicle_id, agent_id, mission, recreate=False, is_hijacked=True
        )
        self.create_vehicle_in_providers(vehicle, agent_id, True)

        return vehicle

    def switch_control_to_agent(
        self,
        vehicle_id: str,
        agent_id: str,
        mission: NavigationMission,
        recreate: bool,
        is_hijacked: bool,
    ) -> Vehicle:
        """Give control of the specified vehicle to the given agent.

        It is not possible to take over a vehicle already controlled by another agent.
        """
        self._check_valid()

        assert not self.vehicle_index.vehicle_is_hijacked(
            vehicle_id
        ), f"Vehicle has already been hijacked: {vehicle_id}"
        assert not vehicle_id in self.vehicle_index.agent_vehicle_ids(), (
            f"`{agent_id}` can't hijack vehicle that is already controlled by an agent"
            f" `{self.agent_manager.agent_for_vehicle(vehicle_id)}`: {vehicle_id}"
        )

        # Switch control to agent
        plan = Plan(self.road_map, mission)
        interface = self.agent_manager.agent_interface_for_agent_id(agent_id)
        self.vehicle_index.start_agent_observation(
            self,
            vehicle_id,
            agent_id,
            interface,
            plan,
            initialize_sensors=not recreate,
        )
        vehicle = self.vehicle_index.switch_control_to_agent(
            self,
            vehicle_id,
            agent_id,
            boid=False,
            recreate=recreate,
            hijacking=is_hijacked,
            agent_interface=interface,
        )

        return vehicle

    def provider_for_actor(self, actor_id: str) -> Optional[Provider]:
        for provider in self.providers:
            if provider.manages_actor(actor_id):
                return provider
        return None

    def _remove_vehicle_from_providers(self, vehicle_id: str):
        for provider in self.providers:
            provider.remove_actor(vehicle_id)

    def create_vehicle_in_providers(
        self,
        vehicle: Vehicle,
        agent_id: str,
        is_ego: bool = False,
    ):
        """Notify providers of the existence of an agent-controlled vehicle,
        one of which should assume management of it."""
        self._check_valid()
        prev_provider: Optional[Provider] = self.provider_for_actor(vehicle.id)
        self._stop_managing_with_providers(vehicle.id)
        role = ActorRole.EgoAgent if is_ego else ActorRole.SocialAgent
        interface = self.agent_manager.agent_interface_for_agent_id(agent_id)
        for provider in self.providers:
            if interface.action in provider.actions:
                state = VehicleState(
                    actor_id=vehicle.id,
                    source=provider.source_str,
                    role=role,
                    vehicle_config_type="passenger",
                    pose=vehicle.pose,
                    dimensions=vehicle.chassis.dimensions,
                    speed=vehicle.speed,
                )
                if provider.can_accept_actor(state):
                    # just use the first provider we find that accepts it
                    # (note that the vehicle will already have a mission plan
                    # registered for it in its sensor state in the vehicle_index.)
                    provider.add_actor(state, prev_provider)
                    return
        # there should always be an AgentsProvider present, so we just assert here
        assert (
            False
        ), f"could not find a provider to accept vehicle {vehicle.id} for agent {agent_id} with role={role.name}"

    def vehicle_exited_bubble(
        self, vehicle_id: str, agent_id: str, teardown_agent: bool
    ):
        """Bubbles call this when a vehicle is exiting the bubble.
        Will try to find a new provider for the vehicle if necessary."""
        original_agent_id = agent_id
        agent_id = None
        # FIXME: This only gets the first shadow agent and this shadow agent is not specific to a bubble!!!!!!
        shadow_agent_id = self._vehicle_index.shadower_id_from_vehicle_id(vehicle_id)
        if shadow_agent_id is not None:
            assert original_agent_id == shadow_agent_id
        if self._vehicle_index.vehicle_is_hijacked(vehicle_id):
            agent_id = self._vehicle_index.owner_id_from_vehicle_id(vehicle_id)
            assert agent_id == original_agent_id
            self._log.debug(
                "agent=%s relinquishing vehicle=%s (shadow_agent=%s)",
                agent_id,
                vehicle_id,
                shadow_agent_id,
            )
            state, route = self._vehicle_index.relinquish_agent_control(
                self, vehicle_id, self.road_map
            )
            new_prov = self._agent_releases_actor(agent_id, state, teardown_agent)
            if (
                route is not None
                and route.road_length > 0
                and isinstance(new_prov, TrafficProvider)
            ):
                new_prov.update_route_for_vehicle(vehicle_id, route)
        if shadow_agent_id:
            self._log.debug(
                "shadow_agent=%s will stop shadowing vehicle=%s",
                shadow_agent_id,
                vehicle_id,
            )
            self._vehicle_index.stop_shadowing(shadow_agent_id, vehicle_id)
            if teardown_agent:
                self.teardown_social_agents([shadow_agent_id])
        if self._vehicle_index.shadower_id_from_vehicle_id(vehicle_id) is None:
            self._sensor_manager.remove_actor_sensors_by_actor_id(vehicle_id)

    def _agent_releases_actor(
        self,
        agent_id: str,
        state: ActorState,
        teardown_agent: bool,
    ) -> Optional[Provider]:
        """Find a new provider for an actor previously managed by an agent.
        Returns the new provider or None if a suitable one could not be found."""
        current_provider = self.provider_for_actor(state.actor_id)
        new_provider = self.provider_releases_actor(current_provider, state)
        if teardown_agent:
            self.teardown_social_agents([agent_id])
        return new_provider

    def provider_relinquishing_actor(
        self, current_provider: Optional[Provider], state: ActorState
    ) -> Tuple[Optional[Provider], ActorProviderTransition]:
        # now try to find one who will take it...
        if isinstance(state, VehicleState):
            state.role = ActorRole.Social  # XXX ASSUMPTION: might use Unknown instead?
        new_provider = None
        for provider in self.providers:
            if provider is current_provider:
                continue
            if provider.can_accept_actor(state):
                # Here we just use the first provider we find that accepts it.
                # If we want to give preference to, say, Sumo over SMARTS traffic,
                # then we should ensure that Sumo comes first in the traffic_sims
                # list we pass to SMARTS __init__().
                new_provider = provider
                break

        actor_transition = ActorProviderTransition()
        object.__setattr__(actor_transition, "current_provider", current_provider)
        object.__setattr__(actor_transition, "actor_state", state)
        return new_provider, actor_transition

    def provider_removing_actor(self, provider: Optional[Provider], actor_id: str):
        # Note: for vehicles, pybullet_provider_sync() will also call teardown
        # when it notices a social vehicle has exited the simulation.
        self._teardown_vehicles([actor_id])

    def _setup_bullet_client(self, client: bc.BulletClient):
        client.resetSimulation()
        client.configureDebugVisualizer(
            pybullet.COV_ENABLE_GUI, 0
        )  # pylint: disable=no-member
        max_pybullet_freq = config()(
            "physics", "max_pybullet_freq", default=MAX_PYBULLET_FREQ, cast=int
        )
        # PyBullet defaults the timestep to 240Hz. Several parameters are tuned with
        # this value in mind. For example the number of solver iterations and the error
        # reduction parameters (erp) for contact, friction and non-contact joints.
        # Attempting to get around this we set the number of substeps so that
        # timestep * substeps = 240Hz. Bullet (C++) does something to this effect as
        # well (https://git.io/Jvf0M), but PyBullet does not expose it.
        # But if our timestep is variable (due to being externally driven)
        # then we will step pybullet multiple times ourselves as necessary
        # to account for the time delta on each SMARTS step.
        self._pybullet_period = (
            self._fixed_timestep_sec
            if self._fixed_timestep_sec
            else 1 / max_pybullet_freq
        )
        client.setPhysicsEngineParameter(
            fixedTimeStep=self._pybullet_period,
            numSubSteps=int(self._pybullet_period * max_pybullet_freq),
            numSolverIterations=10,
            solverResidualThreshold=0.001,
            # warmStartingFactor=0.99
        )

        client.setGravity(0, 0, -9.8)
        self._map_bb = None
        self._setup_pybullet_ground_plane(client)

    def _setup_pybullet_ground_plane(self, client: bc.BulletClient):
        plane_path = self._scenario.plane_filepath
        if not os.path.exists(plane_path):
            with pkg_resources.path(assets, "plane.urdf") as path:
                plane_path = str(path.absolute())

        if not self._map_bb:
            self._map_bb = self.road_map.bounding_box

        if self._map_bb:
            # 1e6 is the default value for plane length and width in smarts/assets/plane.urdf.
            DEFAULT_PLANE_DIM = 1e6
            ground_plane_scale = (
                2.2 * max(self._map_bb.length, self._map_bb.width) / DEFAULT_PLANE_DIM
            )
            ground_plane_center = self._map_bb.center
        else:
            # first step on undefined map, just use a big scale (1e6).
            # it should get updated as soon as vehicles are added...
            ground_plane_scale = 1.0
            ground_plane_center = (0, 0, 0)

        if self._ground_bullet_id is not None:
            client.removeBody(self._ground_bullet_id)
            self._ground_bullet_id = None

        self._ground_bullet_id = client.loadURDF(
            plane_path,
            useFixedBase=True,
            basePosition=ground_plane_center,
            globalScaling=ground_plane_scale,
        )

    def teardown(self):
        """Clean up episode resources."""
        if self._agent_manager is not None:
            self._agent_manager.teardown()
        if self._vehicle_index is not None:
            self._vehicle_index.teardown(self.renderer_ref)
        if self._sensor_manager is not None:
            self._sensor_manager.teardown(self.renderer_ref)

        if self._bullet_client is not None:
            self._bullet_client.resetSimulation()
        if self._renderer is not None:
            self._renderer.teardown()
        self._teardown_providers()

        if self._bubble_manager is not None:
            self._bubble_manager.teardown()
            self._bubble_manager = None
        for actor_capture_manager in self._actor_capture_managers:
            actor_capture_manager.teardown()
        if self._trap_manager is not None:
            self._trap_manager = None

        self._ground_bullet_id = None
        self._is_setup = False

    def destroy(self):
        """Destroy the simulation. Cleans up all remaining simulation resources."""
        if self._is_destroyed:
            return
        self.teardown()

        if self._envision:
            self._envision.teardown()

        if self._visdom:
            self._visdom.teardown()

        if self._agent_manager is not None:
            self._agent_manager.destroy()
            self._agent_manager = None
        if self._vehicle_index is not None:
            self._vehicle_index = None
        for traffic_sim in self._provider_suite.get_all_by_type(TrafficProvider):
            if traffic_sim:
                traffic_sim.destroy()
        self._provider_suite.clear_type(TrafficProvider)
        if self._renderer is not None:
            self._renderer.destroy()
            self._renderer = None
        if self._bullet_client is not None:
            self._bullet_client.disconnect()
            self._bullet_client = None
        self._is_destroyed = True

    def _check_valid(self):
        if self._is_destroyed:
            raise SMARTSDestroyedError(
                "The current SMARTS instance has already been destroyed."
            )

    def __del__(self):
        try:
            self.destroy()
        except (TypeError, AttributeError) as e:
            # This is a print statement because the logging module may be deleted at program exit.
            try:
                exception = SMARTSDestroyedError(
                    "ERROR: A SMARTS instance may have been deleted by gc before a call to destroy."
                    " Please explicitly call `del obj` or `SMARTS.destroy()` to make this error"
                    " go away.",
                    e,
                )
            except (TypeError, KeyboardInterrupt):
                return
            raise exception

    def _teardown_vehicles(self, vehicle_ids):
        self._vehicle_index.teardown_vehicles_by_vehicle_ids(
            vehicle_ids, self.renderer_ref
        )
        self._clear_collisions(vehicle_ids)
        for v_id in vehicle_ids:
            self._remove_vehicle_from_providers(v_id)

    def attach_sensors_to_vehicles(
        self,
        agent_interface: AgentInterface,
        vehicle_ids,
        overwrite_sensors=False,
        reset_sensors=False,
    ):
        """Set the specified vehicles with the sensors needed to satisfy the specified agent
        interface. See :attr:`smarts.core.smarts.SMARTS.prepare_observe_from`.

        Args:
            agent_interface (AgentInterface): The minimum interface generating the observations.
            vehicle_ids (Sequence[str]): The vehicles to target.
            overwrite_sensors (bool, optional): If to replace existing sensors (USE CAREFULLY). Defaults to False.
            reset_sensors (bool, optional): If to remove all existing sensors before adding sensors (USE **VERY** CAREFULLY). Defaults to False.
        """
        self.prepare_observe_from(
            vehicle_ids, agent_interface, overwrite_sensors, reset_sensors
        )

    def prepare_observe_from(
        self,
        vehicle_ids: Sequence[str],
        interface: AgentInterface,
        overwrite_sensors,
        reset_sensors,
    ):
        """Assigns the given vehicle sensors as is described by the agent interface.

        Args:
            vehicle_ids (Sequence[str]): The vehicles to target.
            interface (AgentInterface): The minimum interface generating the observations.
            overwrite_sensors (bool, optional): If to replace existing sensors (USE CAREFULLY).
            reset_sensors (bool, optional): If to remove all existing sensors (USE **VERY** CAREFULLY).
        """
        self._check_valid()

        for v_id in vehicle_ids:
            v = self._vehicle_index.vehicle_by_id(v_id)
            Vehicle.attach_sensors_to_vehicle(
                self.sensor_manager,
                self,
                v,
                interface,
                replace=overwrite_sensors,
                reset_sensors=reset_sensors,
            )

    def observe_from(
        self,
        vehicle_ids: Sequence[str],
        interface: AgentInterface,
    ) -> Tuple[Dict[str, Observation], Dict[str, bool]]:
        """Generate observations from the specified vehicles.

        Args:
            vehicle_ids (Sequence[str]): The vehicles to target.
            interface (AgentInterface): The intended interface generating the observations (this may be ignored.)

        Returns:
            Tuple[Dict[str, Observation], Dict[str, bool]]: A dictionary of observations and the hypothetical dones.
        """
        self._check_valid()

        vehicles: Dict[str, Vehicle] = {
            v_id: self.vehicle_index.vehicle_by_id(v_id) for v_id in vehicle_ids
        }
        sensor_states = {
            vehicle.id: self._sensor_manager.sensor_state_for_actor_id(vehicle.id)
            or SensorState.invalid()
            for vehicle in vehicles.values()
        }
        return self.sensor_manager.observe_batch(
            self.cached_frame,
            self.local_constants,
            interface,
            sensor_states,
            vehicles,
            self.renderer_ref,
            self.bc,
        )

    @property
    def renderer(self) -> RendererBase:
        """The renderer singleton. On call, the sim will attempt to create it if it does not exist."""
        if not self._renderer:
            try:
                from smarts.core.renderer_base import DEBUG_MODE
                from smarts.p3d.renderer import Renderer

                self._renderer = Renderer(self._sim_id, debug_mode=DEBUG_MODE.ERROR)
            except ImportError as exc:
                raise RendererException.required_to("use camera observations") from exc
            except Exception as exc:
                self._renderer = None
                raise RendererException("Unable to create renderer.") from exc
            if not self._renderer.is_setup:
                if self._scenario:
                    self._renderer.setup(self._scenario)
                    self._vehicle_index.begin_rendering_vehicles(self._renderer)
        assert self._renderer is not None
        return self._renderer

    @property
    def renderer_ref(self) -> Optional[RendererBase]:
        """Get the reference of the renderer. This can be `None`."""
        return self._renderer

    @property
    def is_rendering(self) -> bool:
        """If the simulation has image rendering active."""
        return self._renderer is not None

    @property
    def road_stiffness(self) -> Any:
        """The stiffness of the road."""
        return self._bullet_client.getDynamicsInfo(self._ground_bullet_id, -1)[9]

    @property
    def dynamic_action_spaces(self) -> Set[ActionSpaceType]:
        """The set of vehicle action spaces that use dynamics (physics)."""
        return self._agent_physics_provider.actions

    @property
    def traffic_sims(self) -> List[TrafficProvider]:
        """The underlying traffic simulations."""
        return self._provider_suite.get_all_by_type(TrafficProvider)

    @property
    def traffic_history_provider(self) -> TrafficHistoryProvider:
        """The source of any traffic history data."""
        return self._traffic_history_provider

    @property
    def road_map(self) -> RoadMap:
        """The road map api which allows lookup of road features."""
        return self.scenario.road_map

    @property
    def external_provider(self) -> ExternalProvider:
        """The external provider that can be used to inject vehicle states directly."""
        return self._external_provider

    @property
    def bc(self):
        """The bullet physics client instance."""
        return self._bullet_client

    @property
    def envision(self) -> Optional[EnvisionClient]:
        """The envision instance"""
        return self._envision

    @property
    def step_count(self) -> int:
        """The number of steps since the last reset."""
        return self._step_count

    @property
    def elapsed_sim_time(self) -> float:
        """Elapsed time since simulation start."""
        return self._elapsed_sim_time

    @property
    def version(self) -> str:
        """SMARTS version."""
        return VERSION

    def teardown_social_agents(self, agent_ids: Iterable[str]):
        """Tear-down agents in the given sequence.

        Args:
            agent_ids: A sequence of agent ids to terminate and release.
        """
        agents_to_teardown = {
            id_
            for id_ in agent_ids
            if not self.agent_manager.is_boid_agent(id_)
            or self.agent_manager.is_boid_done(id_)
        }
        self.agent_manager.teardown_social_agents(filter_ids=agents_to_teardown)

    def teardown_social_agents_without_actors(self, agent_ids: Iterable[str]):
        """Tear-down agents in the given list that have no actors registered as
        controlled-by or shadowed-by (for each given agent.)

        Args:
            agent_ids: Sequence of agent ids
        """
        self._check_valid()
        original_agents = set(agent_ids)
        agents_to_teardown = {
            agent_id
            for agent_id in original_agents
            # Only clean-up when there is no actor association left
            if len(
                self._vehicle_index.vehicles_by_owner_id(
                    agent_id, include_shadowers=True
                )
            )
            == 0
        }
        if self._log.isEnabledFor(logging.WARNING):
            skipped_agents = original_agents - agents_to_teardown
            if len(skipped_agents) > 0:
                self._log.warning(
                    "Some agents were skipped because they still had vehicles: %s",
                    skipped_agents,
                )

        self.teardown_social_agents(agent_ids=agents_to_teardown)

    def _teardown_vehicles_and_agents(self, vehicle_ids):
        shadow_and_controlling_agents = set()
        for vehicle_id in vehicle_ids:
            agent_id = self._vehicle_index.owner_id_from_vehicle_id(vehicle_id)
            if agent_id:
                shadow_and_controlling_agents.add(agent_id)

            shadow_agent_id = self._vehicle_index.shadower_id_from_vehicle_id(
                vehicle_id
            )
            if shadow_agent_id:
                shadow_and_controlling_agents.add(shadow_agent_id)

        self._vehicle_index.teardown_vehicles_by_vehicle_ids(
            vehicle_ids, self.renderer_ref
        )
        self.teardown_social_agents_without_actors(shadow_and_controlling_agents)
        # XXX: don't remove vehicle from its (traffic) Provider here, as it may be being teleported
        # (and needs to remain registered in Traci during this step).

    def _pybullet_provider_sync(self, provider_state: ProviderState):
        current_actor_ids = {v.actor_id for v in provider_state.actors}
        previous_sv_ids = self._vehicle_index.social_vehicle_ids()
        exited_actors = previous_sv_ids - current_actor_ids
        self._teardown_vehicles_and_agents(exited_actors)

        # Update our pybullet world given this provider state
        dt = provider_state.dt or self._last_dt
        for vehicle in provider_state.actors:
            if not isinstance(vehicle, VehicleState):
                continue
            vehicle_id = vehicle.actor_id
            # Either this is an agent vehicle (ego or social), or it is a social vehicle (traffic).
            # If it's controlled by an agent, then its state will have been updated
            # via perform_agent_actions() already (through an appropriate controller).
            # So here, we just deal with social (traffic) vehicles...
            if vehicle_id not in self._vehicle_index.agent_vehicle_ids():
                if vehicle_id in self._vehicle_index.social_vehicle_ids():
                    social_vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
                else:
                    # It is a new social vehicle we have not seen yet.
                    # Create it's avatar.
                    social_vehicle = self._vehicle_index.build_social_vehicle(
                        sim=self,
                        vehicle_state=vehicle,
                        owner_id="",
                        vehicle_id=vehicle_id,
                    )

                if not vehicle.updated:
                    # Note: update_state() happens *after* pybullet has been stepped.
                    social_vehicle.update_state(vehicle, dt=dt)

    def _step_pybullet(self):
        self._bullet_client.stepSimulation()
        pybullet_substeps = max(1, round(self._last_dt / self._pybullet_period)) - 1
        for _ in range(pybullet_substeps):
            for vehicle in self._vehicle_index.vehicles:
                vehicle.chassis.reapply_last_control()
            self._bullet_client.stepSimulation()
        for vehicle in self._vehicle_index.vehicles:
            vehicle.step(self._elapsed_sim_time)

    @property
    def vehicle_index(self) -> VehicleIndex:
        """The vehicle index for direct vehicle manipulation."""
        return self._vehicle_index

    @property
    def agent_manager(self) -> AgentManager:
        """The agent manager for direct agent manipulation."""
        return self._agent_manager

    @property
    def sensor_manager(self) -> SensorManager:
        """The sensor manager for direct manipulation."""
        return self._sensor_manager

    @property
    def providers(self) -> List[Provider]:
        """The current providers controlling actors within the simulation."""
        return self._provider_suite.instances

    def get_provider_by_type(self, requested_type) -> Optional[Provider]:
        """Get The first provider that matches the requested type."""
        self._check_valid()
        for provider in self.providers:
            if isinstance(provider, requested_type):
                return provider
        return None

    def get_provider_by_id(self, requested_id) -> Optional[Provider]:
        """Find the current provider by its identifier.

        Args:
            requested_id (str): The provider identifier.

        Returns:
            Optional[Provider]: The provider if it exists, otherwise `None`.
        """
        self._check_valid()
        for provider in self.providers:
            if provider.provider_id() == requested_id:
                return provider
        return None

    def _setup_providers(self, scenario) -> ProviderState:
        provider_state = ProviderState()
        for provider in self.providers:
            try:
                new_provider_state = provider.setup(scenario)
            except Exception as provider_error:
                new_provider_state = self._handle_provider(provider, provider_error)
            provider_state.merge(new_provider_state)
        return provider_state

    def _teardown_providers(self):
        for provider in self.providers:
            provider.teardown()
        self._last_provider_state = None

    def _harmonize_providers(self, provider_state: ProviderState):
        for provider in self.providers:
            try:
                provider.sync(provider_state)
            except Exception as provider_error:
                self._handle_provider(provider, provider_error)
        self._pybullet_provider_sync(provider_state)

    def _reset_providers(self):
        for provider in self.providers:
            try:
                provider.reset()
            except Exception as provider_error:
                self._handle_provider(provider, provider_error)

    def _handle_provider(
        self, provider: Provider, provider_error
    ) -> Optional[ProviderState]:
        provider_problem = bool(provider_error or not provider.connected)
        if not provider_problem:
            return None

        recovery_flags = provider.recovery_flags
        recovered = False
        provider_state = None
        if recovery_flags & ProviderRecoveryFlags.ATTEMPT_RECOVERY:
            provider_state, recovered = provider.recover(
                self._scenario, self.elapsed_sim_time, provider_error
            )

        provider_state = provider_state or ProviderState()
        if recovered:
            return provider_state

        if recovery_flags & ProviderRecoveryFlags.RELINQUISH_ACTORS:
            # see if any other provider can take control of its actors...
            self._log.warning(
                "attempting to transfer actors from {provider.source_str} to other providers..."
            )
            for actor in provider_state.actors:
                self.provider_releases_actor(provider, actor)

        if recovery_flags & ProviderRecoveryFlags.EPISODE_REQUIRED:
            self._reset_required = True
            if self._resetting:
                self._log.error(
                    f"`Provider {provider.source_str} has crashed during reset`"
                )
                raise provider_error
            return provider_state
        elif recovery_flags & ProviderRecoveryFlags.EXPERIMENT_REQUIRED:
            raise provider_error

        # default to re-raise error
        raise provider_error

    def _provider_actions(
        self, provider: Provider, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        agent_actions = dict()
        vehicle_actions = dict()
        for agent_id, action in actions.items():
            # TAI:  reconsider include_shadowers = True
            vehicles = self._vehicle_index.vehicles_by_owner_id(
                agent_id, include_shadowers=True
            )
            if not vehicles:
                continue

            interface = self._agent_manager.agent_interface_for_agent_id(agent_id)
            assert interface, f"agent {agent_id} has no interface"
            if interface.action not in provider.actions:
                continue
            assert isinstance(provider, AgentsProvider)

            agent_actions[agent_id] = action

            if self._agent_manager.is_boid_agent(agent_id):
                vehicle_ids = [v.id for v in vehicles]
                for vehicle_id, vehicle_action in action.items():
                    assert vehicle_id in vehicle_ids
                    vehicle_actions[vehicle_id] = vehicle_action
            else:
                assert len(vehicles) == 1
                vehicle_actions[vehicles[0].id] = action

        return agent_actions, vehicle_actions

    def _step_providers(self, actions) -> ProviderState:
        provider_vehicle_actions = dict()
        for provider in self.providers:
            with timeit(
                f"Performing actions on {provider.__class__.__name__}", self._log.debug
            ):
                agent_actions, vehicle_actions = self._provider_actions(
                    provider, actions
                )
                provider_vehicle_actions[provider] = vehicle_actions
                if isinstance(provider, AgentsProvider):
                    provider.perform_agent_actions(agent_actions)

        with timeit("Stepping physics", self._log.debug):
            self._check_ground_plane()
            self._step_pybullet()
            self._process_collisions()

        accumulated_provider_state = ProviderState()

        agent_vehicle_ids = self._vehicle_index.agent_vehicle_ids()
        for provider in self.providers:
            with timeit(f"Stepping {provider.__class__.__name__}", self._log.debug):
                try:
                    provider_state = provider.step(
                        provider_vehicle_actions[provider],
                        self._last_dt,
                        self._elapsed_sim_time,
                    )
                except Exception as provider_error:
                    provider_state = self._handle_provider(provider, provider_error)
                    raise

                # by this point, "stop_managing()" should have been called for the hijacked vehicle on all TrafficProviders
                assert not isinstance(
                    provider, TrafficProvider
                ) or not provider_state.intersects(
                    agent_vehicle_ids
                ), f"{agent_vehicle_ids} in {provider_state.actors}"

                accumulated_provider_state.merge(provider_state)

        with timeit("Harmonizing provider state", self._log.debug):
            self._harmonize_providers(accumulated_provider_state)
        return accumulated_provider_state

    def _sync_smarts_and_provider_actor_states(
        self, external_provider_state: ProviderState
    ):
        self._last_provider_state = external_provider_state
        self._vehicle_states = [v.state for v in self._vehicle_index.vehicles]
        self._last_provider_state.replace_actor_type(self._vehicle_states, VehicleState)

    @property
    def should_reset(self):
        """If the simulation requires a reset."""
        return self._reset_required

    @property
    def resetting(self) -> bool:
        """If the simulation is currently resetting"""
        return self._resetting

    @property
    def scenario(self) -> Scenario:
        """The current simulation scenario."""
        return self._scenario

    @property
    def fixed_timestep_sec(self) -> float:
        """The simulation fixed time-step."""
        # May be None if time deltas are externally driven
        return self._fixed_timestep_sec

    @fixed_timestep_sec.setter
    def fixed_timestep_sec(self, fixed_timestep_sec: float):
        if not fixed_timestep_sec:
            max_pybullet_freq = config()(
                "physics", "max_pybullet_freq", default=MAX_PYBULLET_FREQ, cast=int
            )
            self._rounder = rounder_for_dt(round(1 / max_pybullet_freq, 6))
        else:
            self._rounder = rounder_for_dt(fixed_timestep_sec)
        self._fixed_timestep_sec = fixed_timestep_sec
        self._is_setup = False  # need to re-setup pybullet

    @property
    def last_dt(self) -> float:
        """The last delta time."""
        assert not self._last_dt or self._last_dt > 0
        return self._last_dt

    def neighborhood_vehicles_around_vehicle(
        self, vehicle_id: str, radius: Optional[float] = None
    ) -> List[VehicleState]:
        """Find vehicles in the vicinity of the target vehicle."""
        self._check_valid()

        vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
        return neighborhood_vehicles_around_vehicle(
            vehicle.state, self._vehicle_states, radius
        )

    def _clear_collisions(self, vehicle_ids):
        for vehicle_id in vehicle_ids:
            self._vehicle_collisions.pop(vehicle_id, None)

    def _get_pybullet_collisions(self, vehicle_id: str) -> Set[str]:
        vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
        # We are only concerned with vehicle-vehicle collisions
        return {
            p.bullet_id
            for p in vehicle.chassis.contact_points
            if p.bullet_id != self._ground_bullet_id
        }

    def _process_collisions(self):
        self._vehicle_collisions = dict()

        for vehicle_id in self._vehicle_index.agent_vehicle_ids():
            collidee_bullet_ids = self._get_pybullet_collisions(vehicle_id)
            if not collidee_bullet_ids:
                continue
            vehicle_collisions = self._vehicle_collisions.setdefault(vehicle_id, [])
            for bullet_id in collidee_bullet_ids:
                collidee = self._bullet_id_to_vehicle(bullet_id)
                owner_id = self._vehicle_index.owner_id_from_vehicle_id(collidee.id)
                collision = Collision(
                    collidee_id=collidee.id, collidee_owner_id=owner_id
                )
                vehicle_collisions.append(collision)

        traffic_providers = self._provider_suite.get_all_by_type(TrafficProvider)[:]
        for vehicle_id in self._vehicle_index.social_vehicle_ids():
            for provider in traffic_providers:
                if provider.manages_actor(vehicle_id) and self._get_pybullet_collisions(
                    vehicle_id
                ):
                    provider.vehicle_collided(vehicle_id)

    def _bullet_id_to_vehicle(self, bullet_id):
        for vehicle in self._vehicle_index.vehicles:
            if bullet_id == vehicle.chassis.bullet_id:
                return vehicle
        assert (
            False
        ), f"Only collisions with agent or social vehicles is supported, hit {bullet_id}"

    def _check_ground_plane(self):
        rescale_plane = False
        map_min = np.array(self._map_bb.min_pt)[:2] if self._map_bb else np.array([])
        map_max = np.array(self._map_bb.max_pt)[:2] if self._map_bb else np.array([])
        for vehicle_id in self._vehicle_index.agent_vehicle_ids():
            vehicle = self._vehicle_index.vehicle_by_id(vehicle_id)
            map_spot = vehicle.pose.point.as_np_array[:2]
            if len(map_min) == 0:
                map_min = map_spot
                rescale_plane = True
            elif any(map_spot < map_min):
                map_min = np.minimum(map_spot, map_min)
                rescale_plane = True
            if len(map_max) == 0:
                map_max = map_spot
                rescale_plane = True
            elif any(map_spot > map_max):
                map_max = np.maximum(map_spot, map_max)
                rescale_plane = True
        if rescale_plane:
            MIN_DIM = 500.0
            if map_max[0] - map_min[0] < MIN_DIM:
                map_min[0] -= MIN_DIM
                map_max[0] += MIN_DIM
            if map_max[1] - map_min[1] < MIN_DIM:
                map_min[1] -= MIN_DIM
                map_max[1] += MIN_DIM
            self._map_bb = BoundingBox(Point(*map_min), Point(*map_max))
            self._log.info(
                f"rescaling pybullet ground plane to at least {map_min} and {map_max}"
            )
            self._setup_pybullet_ground_plane(self._bullet_client)

    def _try_emit_envision_state(self, provider_state: ProviderState, obs, scores):
        if not self._envision:
            return

        filter = self._envision.envision_state_filter

        traffic = {}
        signals = dict()
        lane_ids = {}
        agent_vehicle_ids = self._vehicle_index.agent_vehicle_ids()
        vt_mapping = {
            "passenger": envision_types.VehicleType.Car,
            "bus": envision_types.VehicleType.Bus,
            "coach": envision_types.VehicleType.Coach,
            "truck": envision_types.VehicleType.Truck,
            "trailer": envision_types.VehicleType.Trailer,
            "motorcycle": envision_types.VehicleType.Motorcycle,
            "pedestrian": envision_types.VehicleType.Pedestrian,
        }
        for v in provider_state.actors:
            if isinstance(v, SignalState):
                env_ss = envision_types.SignalLightState.Unknown
                if v.state == SignalLightState.OFF:
                    env_ss = envision_types.SignalLightState.Off
                elif v.state & SignalLightState.STOP:
                    env_ss = envision_types.SignalLightState.Stop
                elif v.state & SignalLightState.CAUTION:
                    env_ss = envision_types.SignalLightState.Caution
                elif v.state & SignalLightState.GO:
                    env_ss = envision_types.SignalLightState.Go
                # TODO: eventually do flashing and arrow states too
                signals[v.actor_id] = envision_types.SignalState(
                    v.actor_id, env_ss, tuple(v.stopping_pos.as_np_array)
                )
                continue
            if not isinstance(v, VehicleState):
                continue
            if v.actor_id in agent_vehicle_ids:
                # this is an agent controlled vehicle
                agent_id = self._vehicle_index.owner_id_from_vehicle_id(v.actor_id)
                is_boid_agent = self._agent_manager.is_boid_agent(agent_id)
                agent_obs = obs[agent_id]
                vehicle_obs = agent_obs[v.actor_id] if is_boid_agent else agent_obs
                if (
                    filter.simulation_data_filter["lane_ids"].enabled
                    and vehicle_obs.waypoint_paths
                    and len(vehicle_obs.waypoint_paths[0]) > 0
                ):
                    lane_ids[agent_id] = vehicle_obs.waypoint_paths[0][0].lane_id
                if not filter.simulation_data_filter["traffic"].enabled:
                    continue

                waypoint_paths = []
                if (
                    filter.actor_data_filter["waypoint_paths"].enabled
                    and vehicle_obs.waypoint_paths
                ):
                    waypoint_paths = vehicle_obs.waypoint_paths

                road_waypoints = []
                if (
                    filter.actor_data_filter["road_waypoints"].enabled
                    and vehicle_obs.road_waypoints
                ):
                    road_waypoints = [
                        path
                        for paths in vehicle_obs.road_waypoints.lanes.values()
                        for path in paths
                    ]

                # (points, hits, rays), just want points
                point_cloud = ([], [], [])
                if filter.actor_data_filter["point_cloud"].enabled:
                    point_cloud = (vehicle_obs.lidar_point_cloud or point_cloud)[0]

                # TODO: driven path should be read from vehicle_obs
                driven_path = []
                if filter.actor_data_filter["driven_path"].enabled:
                    driven_path = self._vehicle_index.vehicle_by_id(
                        v.actor_id
                    ).driven_path_sensor(
                        filter.actor_data_filter["driven_path"].max_count
                    )

                mission_route_geometry = None
                if self._agent_manager.is_ego(agent_id):
                    actor_type = envision_types.TrafficActorType.Agent
                    if filter.actor_data_filter["mission_route_geometry"].enabled:
                        mission_route_geometry = (
                            self._sensor_manager.sensor_state_for_actor_id(v.actor_id)
                            .get_plan(self.road_map)
                            .route.geometry
                        )
                else:
                    actor_type = envision_types.TrafficActorType.SocialAgent

                traffic[v.actor_id] = envision_types.TrafficActorState(
                    name=self._agent_manager.agent_name(agent_id),
                    actor_type=actor_type,
                    vehicle_type=envision_types.VehicleType.Car,
                    position=tuple(v.pose.position),
                    heading=float(v.pose.heading),
                    speed=v.speed,
                    actor_id=envision_types.format_actor_id(
                        agent_id,
                        v.actor_id,
                        is_multi=is_boid_agent,
                    ),
                    events=vehicle_obs.events,
                    waypoint_paths=waypoint_paths + road_waypoints,
                    point_cloud=point_cloud,
                    driven_path=driven_path,
                    mission_route_geometry=mission_route_geometry,
                    lane_id=lane_ids.get(agent_id),
                    interest=self.cached_frame.actor_is_interest(v.actor_id),
                )
            elif v.actor_id in self._vehicle_index.social_vehicle_ids():
                # this is a social vehicle
                if filter.simulation_data_filter["traffic"].enabled:
                    veh_type = vt_mapping.get(
                        v.vehicle_config_type
                        if v.vehicle_config_type
                        else v.vehicle_type,
                        envision_types.VehicleType.Car,
                    )
                    traffic[v.actor_id] = envision_types.TrafficActorState(
                        actor_id=v.actor_id,
                        actor_type=envision_types.TrafficActorType.SocialVehicle,
                        vehicle_type=veh_type,
                        position=tuple(v.pose.position),
                        heading=float(v.pose.heading),
                        speed=v.speed,
                        interest=self.cached_frame.actor_is_interest(v.actor_id),
                    )

        bubble_geometry = []
        if filter.simulation_data_filter["bubble_geometry"].enabled:
            bubble_geometry = [
                list(bubble.geometry.exterior.coords)
                for bubble in self._bubble_manager.active_bubbles
            ]

        scenario_folder_path = self.scenario._root
        scenario_name = os.path.split((scenario_folder_path).rstrip("/"))[1]
        assert (
            scenario_name != ""
        ), f"Scenario name was not properly extracted from the scenario folder path: {scenario_folder_path}"

        state = envision_types.State(
            traffic=traffic,
            signals=signals,
            scenario_id=self.scenario.scenario_hash,
            scenario_name=scenario_name,
            bubbles=bubble_geometry,
            scores=scores,
            ego_agent_ids=list(self._agent_manager.ego_agent_ids),
            frame_time=self._rounder(self._elapsed_sim_time + self._total_sim_time),
        )
        self._envision.send(state)

    def _try_emit_visdom_obs(self, obs):
        if not self._visdom:
            return
        self._visdom.send(obs)

    @cached_property
    def local_constants(self):
        """Generate the frozen state that should not change until next reset."""
        self._check_valid()
        road_map, road_map_hash = self.scenario.map_spec.builder_fn(
            self.scenario.map_spec
        )
        return SimulationLocalConstants(
            road_map=road_map,
            road_map_hash=road_map_hash,
        )

    @cached_property
    def cached_frame(self) -> SimulationFrame:
        """Generate a frozen frame state of the simulation."""
        self._check_valid()
        agent_actor_ids = self.vehicle_index.agent_vehicle_ids()
        actor_states = self._last_provider_state
        vehicles = dict(self.vehicle_index.vehicleitems())
        vehicle_ids = set(vid for vid in vehicles)
        return SimulationFrame(
            actor_states=getattr(actor_states, "actors", {}),
            agent_interfaces=self.agent_manager.agent_interfaces.copy(),
            agent_vehicle_controls={
                a_id: self.vehicle_index.owner_id_from_vehicle_id(a_id)
                for a_id in agent_actor_ids
            },
            ego_ids=self.agent_manager.ego_agent_ids,
            pending_agent_ids=self.agent_manager.pending_agent_ids,
            elapsed_sim_time=self.elapsed_sim_time,
            fixed_timestep=self.fixed_timestep_sec,
            resetting=self.resetting,
            # road_map = self.road_map,
            map_spec=self.scenario.map_spec,
            last_dt=self.last_dt,
            last_provider_state=self._last_provider_state,
            step_count=self.step_count,
            vehicle_collisions=self._vehicle_collisions,
            vehicle_states={
                vehicle_state.actor_id: vehicle_state
                for vehicle_state in self._vehicle_states
            },
            vehicles_for_agents={
                agent_id: self.vehicle_index.vehicle_ids_by_owner_id(
                    agent_id, include_shadowers=True
                )
                for agent_id in self.agent_manager.active_agents
            },
            vehicle_ids=vehicle_ids,
            vehicle_sensors=self.sensor_manager.sensors_for_actor_ids(vehicle_ids),
            sensor_states=dict(self.sensor_manager.sensor_states_items()),
            _ground_bullet_id=self._ground_bullet_id,
            interest_filter=re.compile(
                self.scenario.metadata.get("actor_of_interest_re_filter", "")
            ),
        )
