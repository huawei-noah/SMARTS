import logging
import math
from dataclasses import replace
from typing import Sequence, Tuple, Union

from envision.client import Client as Envision
from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Mission, Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.core.utils.math import min_angles_difference_signed

logging.basicConfig(level=logging.INFO)


class PlaceholderAgent(Agent):
    """This is just a place holder such the example code here has a real Agent to work with.
    In actual use, this would be replaced by an agent based on a trained Imitation Learning model."""

    def __init__(self, initial_speed: float = 15.0):
        self._initial_speed = initial_speed
        self._initial_speed_set = False

    @staticmethod
    def _dist(pose1, pose2):
        return math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)

    def act(self, obs: Observation) -> Union[Tuple[float, float], float]:
        if not self._initial_speed_set:
            # special case:  if a singleton int or float is
            # returned, it's taken to be initializing the speed
            self._initial_speed_set = True
            return self._initial_speed

        # Since we don't have a trained model to compute our actions, here we
        # just "fake it" by attempting to match whatever the nearest vehicle
        # is doing...
        if not obs.neighborhood_vehicle_states:
            return (0, 0)
        nn = obs.neighborhood_vehicle_states[0]
        me = obs.ego_vehicle_state
        heading_delta = min_angles_difference_signed(nn.heading, me.heading)
        # ... but if none are nearby, or if the nearest one is going the other way,
        # then we just continue doing whatever we were already doing.
        if (
            PlaceholderAgent._dist(me.position, nn.position) > 15
            or heading_delta > math.pi / 2
        ):
            return (0, 0)
        # simulate a "cushion" we might find in the real data
        avg_following_distance_s = 2
        acceleration = (nn.speed - me.speed) / avg_following_distance_s
        angular_velocity = heading_delta / avg_following_distance_s
        return (acceleration, angular_velocity)


def main(scenarios: Sequence[str], headless: bool, seed: int):
    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=None if headless else Envision(),
    )
    for _ in scenarios:
        scenario = next(scenarios_iterator)
        agent_missions = scenario.discover_missions_of_traffic_histories()

        for agent_id, mission in agent_missions.items():
            agent_spec = AgentSpec(
                interface=AgentInterface.from_type(AgentType.Imitation),
                agent_builder=PlaceholderAgent,
                agent_params=scenario.traffic_history_target_speed,
            )
            agent = agent_spec.build_agent()

            # Take control of vehicle with corresponding agent_id
            smarts.switch_ego_agent({agent_id: agent_spec.interface})

            # tell the traffic history provider to start traffic
            # at the point when this agent enters...
            traffic_history_provider = smarts.get_provider_by_type(
                TrafficHistoryProvider
            )
            assert traffic_history_provider
            traffic_history_provider.start_time = mission.start_time

            # agent vehicle will enter right away...
            modified_mission = replace(mission, start_time=0.0)
            scenario.set_ego_missions({agent_id: modified_mission})

            observations = smarts.reset(scenario)

            dones = {agent_id: False}
            while not dones.get(agent_id, True):
                agent_obs = observations[agent_id]
                agent_action = agent.act(agent_obs)

                observations, rewards, dones, infos = smarts.step(
                    {agent_id: agent_action}
                )

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
    )
