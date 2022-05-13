import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ControllerOutOfLaneException
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)


def _record_data(
    collected_data: List[List[Any]],
    t: float,
    obs: Dict[str, Observation],
    prev_obs: Dict[str, Observation],
):
    for agent_id, car_obs in obs.items():
        curr_state = car_obs.ego_vehicle_state

        dx, dy = None, None
        if prev_obs and agent_id in prev_obs:
            prev_s = prev_obs[agent_id].ego_vehicle_state
            dx = curr_state.position[0] - prev_s.position[0]
            dy = curr_state.position[1] - prev_s.position[1]

        row = [
            t,
            agent_id,
            curr_state.position[0],
            curr_state.position[1],
            dx,
            dy,
            curr_state.speed,
            curr_state.heading,
        ]
        collected_data.append(row)


def main(episodes: int, max_steps: int):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        agent_builder=None,
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
        envision=None,
    )

    scenario_name = "baselines/competition/scenarios/follow"
    scenarios_iterator = Scenario.scenario_variations([scenario_name], [])
    for episode in range(episodes):
        build_scenario([scenario_name])
        scenario = next(scenarios_iterator)
        obs = smarts.reset(scenario)
        prev_obs = obs

        collected_data = []
        _record_data(collected_data, smarts.elapsed_sim_time, obs, None)

        # could also include "motorcycle" or "truck" in this set if desired
        vehicle_types = frozenset({"car"})

        # filter off-road vehicles from observations
        vehicles_off_road = set()

        while smarts.step_count < max_steps:
            smarts.step({})
            current_vehicles = smarts.vehicle_index.social_vehicle_ids(
                vehicle_types=vehicle_types
            )

            if collected_data and not current_vehicles:
                print("no more vehicles.  exiting...")
                break

            for veh_id in current_vehicles:
                try:
                    smarts.attach_sensors_to_vehicles(agent_spec.interface, {veh_id})
                except ControllerOutOfLaneException:
                    logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
                    vehicles_off_road.add(veh_id)

            valid_vehicles = {v for v in current_vehicles if v not in vehicles_off_road}
            obs, _, _, dones = smarts.observe_from(list(valid_vehicles))
            _record_data(collected_data, smarts.elapsed_sim_time, obs, prev_obs)
            prev_obs = obs

        # an example of how we might save the data per car
        observation_folder = "recorded_trajectories"
        if not os.path.exists(observation_folder):
            os.makedirs(observation_folder)

        csv_filename = Path(observation_folder) / f"{scenario.name}-{episode}.csv"
        with open(csv_filename, "w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            header = [
                "sim_time",
                "agent_id",
                "position_x",
                "position_y",
                "delta_x",
                "delta_y",
                "speed",
                "heading",
            ]
            csv_writer.writerow(header)

            for row in collected_data:
                csv_writer.writerow(row)

    smarts.destroy()


if __name__ == "__main__":
    main(episodes=10, max_steps=100)
