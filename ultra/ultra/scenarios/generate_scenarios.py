# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import json
import math
import numpy as np
import os
import random
import shutil
import time
import yaml

from dataclasses import replace
from multiprocessing import Manager, Process
from shutil import copyfile
from smarts.core.utils.sumo import sumolib
from smarts.sstudio import gen_bubbles, gen_missions, gen_traffic
from smarts.sstudio.types import (
    Bubble,
    Flow,
    MapZone,
    Mission,
    PositionalZone,
    Route,
    SocialAgentActor,
    Traffic,
    TrapEntryTactic,
)
from ultra.scenarios.common.begin_time_init_funcs import *
from ultra.scenarios.common.distributions import get_pattern
from ultra.scenarios.common.social_vehicle_definitions import (
    get_social_vehicle_behavior,
)
from typing import Any, Dict, Sequence


LANE_LENGTH = 137.85


def ego_mission_config_to_route(
    ego_mission_config: Dict[str, Any],
    route_lanes: Dict[str, int],
    stopwatcher_behavior: bool,
) -> Route:
    """Creates a Route from an ego mission config dictionary.

    Args:
        ego_mission_config: A dictionary describing the ego mission.
        route_lanes: A dictionary of routes (edges) as keys and the number of lanes in
            each route as values.
        stopwatcher_behavior: A boolean value describing the presence of stopwatchers.

    Returns:
        Route: The route object created from the ego mission config.
    """
    if stopwatcher_behavior:  # Put the ego vehicle(s) on the side road.
        mission_start = "edge-south-side"
        mission_end = "edge-dead-end"
        mission_start_lane_index = 0
        mission_end_lane_index = 0
        mission_start_offset = 100
        mission_end_offset = 5
    else:
        mission_start = "edge-{}".format(ego_mission_config["start"])
        mission_end = "edge-{}".format(ego_mission_config["end"])
        mission_start_lane_index = route_lanes[ego_mission_config["start"]] - 1
        mission_end_lane_index = route_lanes[ego_mission_config["end"]] - 1
        mission_start_offset = (
            random.randint(
                ego_mission_config["start_offset"][0],
                ego_mission_config["start_offset"][1],
            )
            if "start_offset" in ego_mission_config
            else random.randint(50, 120)  # The default range of the offset.
        )
        mission_end_offset = (
            random.randint(
                ego_mission_config["end_offset"][0], ego_mission_config["end_offset"][1]
            )
            if "end_offset" in ego_mission_config
            else random.randint(50, 120)  # The default range of the offset.
        )

    route = Route(
        begin=(
            mission_start,
            mission_start_lane_index,
            mission_start_offset,
        ),
        end=(
            mission_end,
            mission_end_lane_index,
            mission_end_offset,
        ),
    )
    return route


def bubble_config_to_bubble_object(
    scenario: str, bubble_config: Dict[str, Any], vehicles_to_not_hijack: Sequence[str]
) -> Bubble:
    """Converts a bubble config to a bubble object.

    Args:
        scenario:
            A string representing the path to this scenario.
        bubble_config:
            A dictionary with 'location', 'actor_name', 'agent_locator', and
            'agent_params' keys that is used to initialize the bubble.
        vehicles_to_not_hijack:
            A tuple of vehicle IDs that are passed to the bubble. The bubble will not
            capture those vehicles that have an ID in this tuple.

    Returns:
        Bubble: The bubble object created from the bubble config.
    """
    BUBBLE_MARGIN = 2
    map_file = sumolib.net.readNet(f"{scenario}/map.net.xml")

    location_name = bubble_config["location"][0]
    location_data = bubble_config["location"][1:]
    actor_name = bubble_config["actor_name"]
    agent_locator = bubble_config["agent_locator"]
    agent_params = bubble_config["agent_params"]

    if location_name == "intersection":
        # Create a bubble centered at the intersection.
        assert len(location_data) == 2
        bubble_length, bubble_width = location_data
        bubble_coordinates = map_file.getNode("junction-intersection").getCoord()
        zone = PositionalZone(
            pos=bubble_coordinates, size=(bubble_length, bubble_width)
        )
    else:
        # Create a bubble on one of the lanes.
        assert len(location_data) == 4
        lane_index, lane_offset, bubble_length, num_lanes_spanned = location_data
        zone = MapZone(
            start=("edge-" + location_name, lane_index, lane_offset),
            length=bubble_length,
            n_lanes=num_lanes_spanned,
        )

    bubble = Bubble(
        zone=zone,
        actor=SocialAgentActor(
            name=actor_name,
            agent_locator=agent_locator,
            policy_kwargs=agent_params,
            initial_speed=None,
        ),
        margin=BUBBLE_MARGIN,
        limit=None,
        exclusion_prefixes=tuple(vehicles_to_not_hijack),
        follow_actor_id=None,
        follow_offset=None,
        keep_alive=False,
    )
    return bubble


def add_stops_to_traffic(
    scenario: str, stops: Sequence[Sequence[Any]], vehicles_to_not_hijack: Sequence[str]
):
    """Adds stopped vehicles to the traffic by overwriting all.rou.xml and replacing
    some vehicles' attributes so that they start, and remain stopped.

    Args:
        scenario:
            A string representing the path to this scenario.
        stops:
            A list of lists, where each list element contains information about where
            to stop the vehicle. Each element of stops is a list in the form of
            [stop_edge, stop_lane_index, stop_offset]. For stops of length n, n vehicles
            will be chosen to be stopped in the scenario.
        vehicles_to_not_hijack:
            A list of vehicle IDs that is appended to. Each stopped vehicle's ID is
            appended to this list as stopped vehicles should not be hijacked.
    """
    route_file_path = f"{scenario}/traffic/all.rou.xml"
    map_file = sumolib.net.readNet(f"{scenario}/map.net.xml")
    vehicle_types = list(sumolib.output.parse(route_file_path, "vType"))
    vehicles = list()
    stops_added = 0

    # XXX: Stops will not be added to vehicles if they don't match the route and lane of
    #      the vehicle. For each vehicle, we are NOT checking each available stop to see
    #      if the vehicle matches the route and lane, only the stop at the stops_added
    #      index.

    # Add stops (if applicable) to vehicles in the existing all.rou.xml file.
    for vehicle in sumolib.output.parse(route_file_path, "vehicle"):
        if stops_added >= len(stops):
            break

        vehicle_edges = vehicle.route[0].edges.split()
        start_edge = map_file.getEdge(vehicle_edges[0])

        for lane in start_edge.getLanes():
            stop_route, stop_lane, stop_position = stops[stops_added]
            if lane.getID() == f"edge-{stop_route}_{stop_lane}":
                # Add stop information to this vehicle.
                stop_attributes = {
                    "lane": lane.getID(),
                    "endPos": stop_position,
                    "duration": "1000",
                }
                vehicle.setAttribute("depart", 0)
                vehicle.setAttribute("departPos", stop_position)
                vehicle.setAttribute("departSpeed", 0)
                vehicle.setAttribute("departLane", stop_lane)
                vehicle.addChild("stop", attrs=stop_attributes)
                vehicles_to_not_hijack.append(vehicle.id)
                stops_added += 1
                break
        vehicles.append([float(vehicle.depart), vehicle.id, vehicle])
    vehicles.sort(key=lambda x: (x[0], x[1]))

    # Ensure all stops were added to the traffic.
    if stops_added < len(stops):
        print(f"{scenario} has only placed {stops_added} out of {len(stops)} stops.")

    # Overwrite the all.rou.xml file with the new vehicles.
    with open(route_file_path, "w") as route_file:
        sumolib.writeXMLHeader(route_file, "routes")

        route_file.write(
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        )

        for vehicle_type in vehicle_types:
            route_file.write(vehicle_type.toXML(" " * 4))

        for _, _, vehicle in vehicles:
            route_file.write(vehicle.toXML(" " * 4))

        route_file.write("</routes>\n")


def copy_map_files(scenario, map_dir, speed):
    if not os.path.exists(scenario):
        os.makedirs(scenario)
    copyfile(map_dir + f"/{speed}/map.net.xml", scenario + "/map.net.xml")
    copyfile(map_dir + f"/{speed}/map.glb", scenario + "/map.glb")


def get_direction(route_key):
    route_direction = {
        "north-south": ["north-NS", "south-NS"],
        "south-north": ["south-SN", "north-SN"],
        "south-east": ["south-SN", "east-WE"],
        "south-west": ["south-SN", "west-EW"],
        "north-east": ["north-NS", "east-WE"],
        "north-west": ["north-NS", "west-EW"],
        "north-south": ["north-NS", "south-NS"],
        "east-north": ["east-EW", "north-SN"],
        "east-south": ["east-EW", "south-NS"],
        "east-west": ["east-EW", "west-EW"],
        "west-north": ["west-WE", "north-SN"],
        "west-south": ["west-WE", "south-NS"],
        "west-east": ["west-WE", "east-WE"],
    }
    return route_direction[route_key]


def generate_stopwatcher(
    stopwatcher_behavior, stopwatcher_route, start_lane_id, end_lane_id, begin_time=None
):
    # this is a vehicle with aggressive behavior that records the number of steps to finishe a left-turn
    behavior = replace(
        get_social_vehicle_behavior(stopwatcher_behavior),
        name=f"stopwatcher_{stopwatcher_behavior}",
    )
    begin_time = (
        begin_time if begin_time is not None else random.randint(10, 15)
    )  # slight delay to wait for trafficne_id
    return Flow(
        begin=begin_time,
        end=begin_time + 3600,  # 1 hour
        route=Route(
            begin=(
                f"edge-{stopwatcher_route[0]}",
                start_lane_id,
                "base",
            ),
            end=(
                f"edge-{stopwatcher_route[1]}",
                end_lane_id,
                "max",
            ),
        ),
        rate=1,
        actors={behavior: 1.0},
    )


def generate_left_turn_missions(
    missions,
    shuffle_missions,
    route_distributions,
    route_lanes,
    speed,
    map_dir,
    level_name,
    save_dir,
    stopwatcher_behavior,
    stopwatcher_route,
    seed,
    stops,
    bubbles,
    intersection_name,
    traffic_density,
):
    # By default the sumo_seed is set to the scenario seed
    sumo_seed = seed
    stopwatcher_info = None
    stopwatcher_added = False
    if stopwatcher_behavior:
        stopwatcher_info = {
            "behavior": stopwatcher_behavior,
            "direction": get_direction(stopwatcher_route),
        }

    random.seed(seed)
    np.random.seed(seed)
    all_flows = []
    metadata = {"routes": {}, "total_vehicles": 0, "stopwatcher": None}

    scenario = save_dir + f"-flow-{seed}"

    # Remove old traffic route if it exists(otherwise, won't update to new flows)
    if os.path.exists(f"{scenario}/traffic/all.rou.xml"):
        shutil.rmtree(scenario)

    for route_key, route_info in route_distributions["routes"].items():
        # to skip None
        if route_info:
            ego_routes = [
                ego_mission_config_to_route(
                    ego_mission_config=ego_mission_config,
                    route_lanes=route_lanes,
                    stopwatcher_behavior=stopwatcher_behavior,
                )
                for ego_mission_config in missions
            ]
            # Not all routes need to have a custom start/end offset
            if "pos_offsets" in route_info:
                pos_offsets = route_info["pos_offsets"]
            else:
                pos_offsets = None
            flows, vehicles_log_info = generate_social_vehicles(
                route_distribution=route_info["distribution"],
                begin_time_init=route_info["begin_time_init"],
                num_vehicles=route_info["vehicles"],
                route_direction=get_direction(route_key),
                route_lanes=route_lanes,
                route_has_turn=route_info["has_turn"],
                start_end_on_different_lanes_probability=route_info[
                    "start_end_on_different_lanes_probability"
                ],
                stops=stops,
                deadlock_optimization=route_info["deadlock_optimization"],
                pos_offsets=pos_offsets,
                stopwatcher_info=stopwatcher_info,
                traffic_params={"speed": speed, "traffic_density": traffic_density},
            )
            if (
                stopwatcher_behavior
                and len(flows) > 0
                and get_direction(route_key) == stopwatcher_info["direction"]
            ):
                stopwatcher_added = True
                print(
                    f'stop watcher added to {get_direction(route_key)} flows among {route_info["vehicles"]} vehicles!'
                )
            all_flows.extend(flows)
            metadata["routes"][route_key] = vehicles_log_info
    scenario = save_dir + f"-flow-{seed}"
    if stopwatcher_behavior:
        if not stopwatcher_added and level_name != "no-traffic":
            print(
                f'There was no matching flows for stopwatcher, adding it to {stopwatcher_info["direction"]}.'
            )
            # vehicles_log_info[f'stopwatcher_{stopwatcher_info["behavior"]}']["count"] += 1
            all_flows.append(
                generate_stopwatcher(
                    stopwatcher_behavior=stopwatcher_info["behavior"],
                    stopwatcher_route=stopwatcher_info["direction"],
                    start_lane_id=0,
                    end_lane_id=0,
                )
            )
        scenario += f"-stopwatcher-{stopwatcher_info['behavior']}"

    copy_map_files(scenario, map_dir, speed)

    vehicles_to_not_hijack = []
    traffic = Traffic(flows=all_flows)
    try:
        gen_traffic(scenario, traffic, name=f"all", seed=sumo_seed)
        if stops:
            add_stops_to_traffic(scenario, stops, vehicles_to_not_hijack)
    except Exception as exception:
        print(exception)

    # Patch: Remove route files from traffic folder to make intersection empty.
    if traffic_density == "no-traffic":
        os.remove(f"{scenario}/traffic/all.rou.xml")

    if stopwatcher_behavior or "ego_hijacking_params" not in route_distributions:
        mission_objects = [Mission(ego_route) for ego_route in ego_routes]
    else:
        speed_m_per_s = float("".join(filter(str.isdigit, speed))) * 5.0 / 18.0
        hijacking_params = route_distributions["ego_hijacking_params"]
        zone_range = hijacking_params["zone_range"]

        wait_to_hijack_limit_s = hijacking_params["wait_to_hijack_limit_s"]
        waiting_time = (
            np.random.randint(
                wait_to_hijack_limit_s[0],
                wait_to_hijack_limit_s[1],
            )
            if isinstance(wait_to_hijack_limit_s, (list, tuple))
            else wait_to_hijack_limit_s
        )

        start_time = (
            hijacking_params["start_time"]
            if hijacking_params["start_time"] != "default"
            else random.randint((LANE_LENGTH // speed_m_per_s), 60)
        )
        mission_objects = [
            Mission(
                ego_route,
                # Optional: control hijacking time, place, and emission.
                start_time=start_time,  # When to start hijacking (might start later).
                entry_tactic=TrapEntryTactic(
                    wait_to_hijack_limit_s=waiting_time,  # When to give up hijacking.
                    zone=MapZone(
                        start=(
                            ego_route.begin[0],
                            0,
                            ego_route.begin[2] + zone_range[0],
                        ),
                        length=zone_range[1],
                        n_lanes=(ego_route.begin[1] + 1),
                    ),  # Area to hijack.
                    exclusion_prefixes=tuple(vehicles_to_not_hijack),  # Don't hijack.
                ),
            )
            for ego_route in ego_routes
        ]
    # Shuffle the missions so agents don't do the same route all the time.
    if shuffle_missions:
        random.shuffle(mission_objects)
    gen_missions(scenario, mission_objects)

    if bubbles:
        bubble_objects = [
            bubble_config_to_bubble_object(
                scenario, bubble_config, vehicles_to_not_hijack
            )
            for bubble_config in bubbles
        ]
        gen_bubbles(scenario, bubble_objects)

    if stopwatcher_behavior:
        metadata["stopwatcher"] = {
            "direction": stopwatcher_info["direction"],
            "behavior": stopwatcher_info["behavior"],
        }
    metadata["intersection"] = {
        "type": intersection_name[-1],
        "name": intersection_name,
    }
    metadata["total_vehicles"] = len(all_flows)
    metadata["seed"] = seed
    metadata["flow_id"] = f"flow_{seed}"
    with open(f"{scenario}/metadata.json", "w") as log:
        json.dump(metadata, log)
    # print(f"Finished:{scenario}")


def generate_social_vehicles(
    route_distribution,
    start_end_on_different_lanes_probability,
    num_vehicles,
    route_direction,
    route_lanes,
    route_has_turn,
    stopwatcher_info,
    traffic_params,
    stops,
    pos_offsets,
    begin_time_init=None,
    deadlock_optimization=True,
):
    flows = []
    behaviors = []
    log_info = {
        "num_vehicles": 0,
        "route_distribution": None,
        "start_end_on_different_lanes_probability": 0.0,
    }
    stopwatcher_added = False
    # populate random routes based on their probability
    start_lane, end_lane = route_direction
    if stopwatcher_info and route_direction == stopwatcher_info["direction"]:
        if stopwatcher_info["behavior"] not in behaviors:
            behaviors.append(f'stopwatcher_{stopwatcher_info["behavior"]}')
            num_vehicles = max(0, num_vehicles - 1)  # used 1 for stopwatcher
            stopwatcher_added = True

    behaviors.extend(
        random.choices(
            list(route_distribution.keys()),
            weights=list(route_distribution.values()),
            k=num_vehicles,
        )
    )

    random.shuffle(behaviors)  # because stopwatcher is always at the beginning

    if begin_time_init is None:
        begin_time_init_func = basic_begin_time_init_func
        begin_time_init_params = {"probability": 0.0}
    else:
        begin_time_init_func = begin_time_init["func"]
        begin_time_init_params = begin_time_init["params"]

    begin_time_init_params = (
        {} if begin_time_init_params is None else begin_time_init_params
    )
    begin_times = begin_time_init_func(
        route_lanes[start_lane],
        len(behaviors),
        traffic_params,
        **begin_time_init_params,
    )
    begin_time_idx = [0 for _ in range(route_lanes[start_lane])]

    for behavior_idx in behaviors:
        if behavior_idx not in log_info:
            log_info[behavior_idx] = {"count": 0, "start_end_different_lanes": 0}
        if deadlock_optimization and route_has_turn:
            # if route has a turn, start on the left-most lane
            start_lane_id = route_lanes[start_lane] - 1
            lane_changing_options = [
                lane_id for lane_id in range(route_lanes[end_lane])
            ]
            if (
                len(lane_changing_options) > 0
                or random.uniform(0, 1) < start_end_on_different_lanes_probability
            ):
                end_lane_id = random.choice(lane_changing_options)
            else:
                end_lane_id = start_lane_id

        else:
            start_lane_id = random.randint(0, route_lanes[start_lane] - 1)
            end_lane_id = random.randint(0, route_lanes[end_lane] - 1)

        # set begin/end time
        begin_time = begin_times[start_lane_id][begin_time_idx[start_lane_id]]
        begin_time_idx[start_lane_id] += 1
        end_time = begin_time + 3600  # 1 hour

        if "stopwatcher" in behavior_idx:
            start_lane_id = route_lanes[stopwatcher_info["direction"][0]] - 1
            end_lane_id = route_lanes[stopwatcher_info["direction"][1]] - 1
            # To ensure that the stopwatcher spawns in all scenarios the
            # stopwatcher's begin time is bounded between 10s to 50s
            # (100ts to 500ts, if 1s = 10 ts). During analysis, the
            # stopwatcher is guaranteed to spawn before the 500ts
            # and no less then 100ts
            begin_time = random.randint(10, 50)
            flows.append(
                generate_stopwatcher(
                    stopwatcher_behavior=stopwatcher_info["behavior"],
                    stopwatcher_route=stopwatcher_info["direction"],
                    begin_time=begin_time,
                    start_lane_id=start_lane_id,
                    end_lane_id=end_lane_id,
                )
            )
        else:
            behavior = get_social_vehicle_behavior(behavior_idx)
            if pos_offsets:
                start_offset = random.randint(
                    pos_offsets["start"][0], pos_offsets["start"][1]
                )
                end_offset = random.randint(
                    pos_offsets["end"][0], pos_offsets["end"][1]
                )
            else:
                start_offset = "base"
                end_offset = "max"
            flows.append(
                Flow(
                    begin=begin_time,
                    end=end_time,
                    route=Route(
                        begin=(f"edge-{start_lane}", start_lane_id, start_offset),
                        end=(f"edge-{end_lane}", end_lane_id, end_offset),
                    ),
                    rate=1,
                    actors={behavior: 1.0},
                )
            )
        log_info[behavior_idx]["count"] += 1
        log_info["route_distribution"] = route_distribution
        log_info["num_vehicles"] = (
            num_vehicles + 1 if stopwatcher_added else num_vehicles
        )
        log_info[
            "start_end_on_different_lanes_probability"
        ] = start_end_on_different_lanes_probability
        log_info[behavior_idx]["start_end_different_lanes"] += (
            1 if start_lane_id != end_lane_id else 0
        )

    return flows, log_info


def scenario_worker(
    seeds,
    ego_missions,
    shuffle_missions,
    route_lanes,
    route_distributions,
    map_dir,
    stopwatcher_route,
    level_name,
    save_dir,
    speed,
    stopwatcher_behavior,
    traffic_density,
    intersection_type,
    mode,
    total_seeds,
    percent,
    stops,
    bubbles,
    dynamic_pattern_func,
):
    for i, seed in enumerate(seeds):
        if not dynamic_pattern_func is None:
            route_distributions = dynamic_pattern_func(route_distributions, i)

        generate_left_turn_missions(
            missions=ego_missions,
            shuffle_missions=shuffle_missions,
            route_lanes=route_lanes,
            route_distributions=route_distributions,
            map_dir=map_dir,
            level_name=level_name,
            save_dir=save_dir,
            speed=speed,
            stopwatcher_behavior=stopwatcher_behavior,
            stopwatcher_route=stopwatcher_route,
            seed=seed,
            stops=stops,
            bubbles=bubbles,
            traffic_density=traffic_density,
            intersection_name=intersection_type,
        )
    # print(
    #     f"{mode} {intersection_type} {speed} {traffic_density}, counts:{len(seeds)}, generated:{len(seeds)/len(total_seeds)}, real:{percent}"
    # )


def build_scenarios(
    task,
    level_name,
    stopwatcher_behavior,
    stopwatcher_route,
    save_dir,
    root_path,
    shuffle_missions=True,
    seed=None,
    pool_dir=None,
    dynamic_pattern_func=None,
):
    print("Generating Scenario ...")
    manager = Manager()

    log_dict = manager.dict()

    with open(f"{root_path}/{task}/config.yaml", "r") as task_file:
        task_config = yaml.safe_load(task_file)
        print(f"{root_path}/{task}/config.yaml")

    level_config = task_config["levels"][level_name]
    scenarios_dir = os.path.dirname(os.path.realpath(__file__))
    task_dir = f"{scenarios_dir}/{task}"
    pool_dir = f"{scenarios_dir}/pool/experiment_pool" if pool_dir is None else pool_dir

    train_total = int(level_config["train"]["total"])
    test_total = int(level_config["test"]["total"])
    if seed is None:
        # Generate seeds 0, 1, ..., train_total + test_total - 1, and allocate the
        # first train_total seeds to the training scenarios, and the rest to the testing
        # scenarios.
        scenario_seeds = [i for i in range(train_total + test_total)]
    else:
        # Generate random seeds for the scenarios by sampling numbers in the range
        # [0, 2** 31) without replacement. The generation of these seeds is seeded by
        # the seed passed to this function.
        _seeded_random = random.Random(seed)
        scenario_seeds = _seeded_random.sample(range(2 ** 31), train_total + test_total)
    splitted_seeds = {
        "train": scenario_seeds[:train_total],
        "test": scenario_seeds[train_total : (train_total + test_total)],
    }

    jobs = []
    start = time.time()
    for mode, mode_seeds in splitted_seeds.items():
        combinations = []

        # Obtain the ego missions specified for this mode and ensure
        # that test scenarios only have one ego mission.
        ego_missions = level_config[mode]["ego_missions"]
        assert not (
            mode == "test" and (len(ego_missions) != 1)
        ), "Test scenarios must have one ego mission."

        prev_split = 0
        main_seed_count = 0
        # sort inverse by percents
        intersection_types = level_config[mode]["intersection_types"]
        intersections = sorted(
            [
                [
                    _type,
                    intersection_types[_type]["percent"],
                    intersection_types[_type]["stops"],
                    intersection_types[_type]["bubbles"],
                ]
                for _type in intersection_types
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        log_dict[mode] = {x: {"count": 0, "percent": 0} for x in intersection_types}
        for intersection_type, intersection_percent, stops, bubbles in intersections:
            part = int(float(intersection_percent) * len(mode_seeds))
            cur_split = prev_split + part
            seeds = mode_seeds[prev_split:cur_split]
            specs = sorted(
                intersection_types[intersection_type]["specs"],
                key=lambda x: float(x[2]),
                reverse=True,
            )
            seed_count = 0
            map_dir = f"{pool_dir}/{intersection_type}"
            with open(f"{map_dir}/info.json") as jsonfile:
                map_metadata = json.load(jsonfile)
                route_lanes = map_metadata["num_lanes"]
            inner_prev_split = 0
            for speed, traffic_density, percent in specs:
                inner_part = math.ceil(float(percent) * len(seeds))

                inner_cur_split = inner_prev_split + inner_part
                name_additions = [mode, task, level_name, intersection_type, speed]

                if level_name != "no-traffic":
                    name_additions.append(traffic_density)

                route_distributions = get_pattern(traffic_density, intersection_type)
                temp_seeds = seeds[inner_prev_split:inner_cur_split]
                seed_count += len(temp_seeds)
                if save_dir is None:
                    temp_save_dir = os.path.join(task_dir, "_".join(name_additions))
                else:
                    temp_save_dir = os.path.join(save_dir, "_".join(name_additions))

                sub_proc = Process(
                    target=scenario_worker,
                    args=(
                        temp_seeds,
                        ego_missions,
                        shuffle_missions,
                        route_lanes,
                        route_distributions,
                        map_dir,
                        stopwatcher_route,
                        level_name,
                        temp_save_dir,
                        speed,
                        stopwatcher_behavior,
                        traffic_density,
                        intersection_type,
                        mode,
                        seeds,
                        percent,
                        stops,
                        bubbles,
                        dynamic_pattern_func,
                    ),
                )
                jobs.append(sub_proc)
                sub_proc.start()
                inner_prev_split = inner_cur_split
            generated = seed_count / len(mode_seeds) if len(mode_seeds) > 0 else 0
            generation_stats = (
                f">> {mode} {intersection_type} "
                f"count: {seed_count} "
                f"generated: {generated} "
                f"real: {intersection_percent}"
            )
            print(generation_stats)
            prev_split = cur_split
            main_seed_count += seed_count
    for process in jobs:
        process.join()
    print("*** time took:", time.time() - start)
