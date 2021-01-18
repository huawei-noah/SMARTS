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
import os, argparse, json, shutil
from multiprocessing import Process, Manager
import itertools, random, copy
from dataclasses import replace
from smarts.sstudio.types import Route
import yaml, json, copy, math
import numpy as np
from collections import defaultdict, Counter
import random, os, json, time
from shutil import copyfile
import numpy as np
from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    Distribution,
    Mission,
    MapZone,
    TrapEntryTactic,
)
from ultra.scenarios.common.distributions import get_pattern
from ultra.scenarios.common.social_vehicle_definitions import (
    get_social_vehicle_behavior,
)
from ultra.scenarios.common.begin_time_init_funcs import *

LANE_LENGTH = 137.85


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
            begin=(f"edge-{stopwatcher_route[0]}", start_lane_id, "base",),
            end=(f"edge-{stopwatcher_route[1]}", end_lane_id, "max",),
        ),
        rate=1,
        actors={behavior: 1.0},
    )


def generate_left_turn_missions(
    mission,
    route_distributions,
    route_lanes,
    speed,
    map_dir,
    level_name,
    save_dir,
    stopwatcher_behavior,
    stopwatcher_route,
    seed,
    intersection_name,
    traffic_density,
):
    # dont worry about these seeds, theyre used by sumo
    sumo_seed = random.choice([0, 1, 2, 3, 4])
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
            if stopwatcher_behavior:  # put the ego on the side road
                ego_start_lane, ego_end_lane = 0, 0
                mission_start, mission_end = "south-side", "dead-end"
                ego_start_pos, ego_end_pos = 100, 5
                ego_num_lanes = 1
            else:
                mission_start, mission_end = mission["start"], mission["end"]
                ego_start_lane, ego_end_lane = (
                    route_lanes[mission_start] - 1,
                    route_lanes[mission_end] - 1,
                )
                ego_start_pos, ego_end_pos = (
                    random.randint(50, 120),
                    random.randint(50, 150),
                )
                ego_num_lanes = route_lanes[mission_start]

            ego_route = Route(
                begin=(f"edge-{mission_start}", ego_start_lane, ego_start_pos),
                end=(f"edge-{mission_end}", ego_end_lane, ego_end_pos),
            )
            flows, vehicles_log_info = generate_social_vehicles(
                ego_start_lane=ego_start_lane,
                route_distribution=route_info["distribution"],
                begin_time_init=route_info["begin_time_init"],
                num_vehicles=route_info["vehicles"],
                route_direction=get_direction(route_key),
                route_lanes=route_lanes,
                route_has_turn=route_info["has_turn"],
                start_end_on_different_lanes_probability=route_info[
                    "start_end_on_different_lanes_probability"
                ],
                deadlock_optimization=route_info["deadlock_optimization"],
                stopwatcher_info=stopwatcher_info,
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
    if stopwatcher_behavior or "ego_hijacking_params" not in route_distributions:
        gen_missions(scenario, [Mission(ego_route)])
    else:
        speed_m_per_s = float("".join(filter(str.isdigit, speed))) * 5.0 / 18.0
        hijacking_params = route_distributions["ego_hijacking_params"]
        zone_range = hijacking_params["zone_range"]
        waiting_time = hijacking_params["wait_to_hijack_limit_s"]
        start_time = hijacking_params["start_time"]

        if start_time == "default":
            start_time = random.randint((LANE_LENGTH // speed_m_per_s), 60)
        gen_missions(
            scenario,
            [
                Mission(
                    ego_route,
                    # optional: control hijacking time, place, and emission.
                    start_time=start_time,  # when to start hijacking (might start later)
                    entry_tactic=TrapEntryTactic(
                        wait_to_hijack_limit_s=waiting_time,  # when to give up on hijacking and start emitting a social vehicle instead
                        zone=MapZone(
                            start=(
                                f'edge-{mission["start"]}',
                                0,
                                ego_start_pos + zone_range[0],
                            ),
                            length=zone_range[1],
                            n_lanes=route_lanes[mission["start"]],
                        ),  # area to hijack
                        exclusion_prefixes=tuple(),  # vehicles to be excluded (check vehicle ids)
                    ),
                ),
            ],
        )

    traffic = Traffic(flows=all_flows)
    gen_traffic(scenario, traffic, name=f"all", seed=sumo_seed)
    # patch: remove route files from traffic folder to make intersection empty
    if traffic_density == "no-traffic":

        os.remove(f"{scenario}/traffic/all.rou.xml")
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
    ego_start_lane,
    route_distribution,
    start_end_on_different_lanes_probability,
    num_vehicles,
    route_direction,
    route_lanes,
    route_has_turn,
    stopwatcher_info,
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
        route_lanes[start_lane], len(behaviors), **begin_time_init_params
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
            flows.append(
                Flow(
                    begin=begin_time,
                    end=end_time,
                    route=Route(
                        begin=(f"edge-{start_lane}", start_lane_id, "base"),
                        end=(f"edge-{end_lane}", end_lane_id, "max"),
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
    ego_mission,
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
    dynamic_pattern_func,
):
    for i, seed in enumerate(seeds):
        if not dynamic_pattern_func is None:
            route_distributions = dynamic_pattern_func(route_distributions, i)

        generate_left_turn_missions(
            mission=ego_mission,
            route_lanes=route_lanes,
            route_distributions=route_distributions,
            map_dir=map_dir,
            level_name=level_name,
            save_dir=save_dir,
            speed=speed,
            stopwatcher_behavior=stopwatcher_behavior,
            stopwatcher_route=stopwatcher_route,
            seed=seed,
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
    dynamic_pattern_func=None,
):
    print("Generating Scenario ...")
    manager = Manager()

    log_dict = manager.dict()

    with open(f"{root_path}/{task}/config.yaml", "r") as task_file:
        task_config = yaml.safe_load(task_file)
        print(f"{root_path}/{task}/config.yaml")

    ego_mission = task_config["ego_mission"]
    level_config = task_config["levels"][level_name]
    scenarios_dir = os.path.dirname(os.path.realpath(__file__))
    task_dir = f"{scenarios_dir}/{task}"
    pool_dir = f"{scenarios_dir}/pool"

    train_total, test_total = (
        int(level_config["train"]["total"]),
        int(level_config["test"]["total"]),
    )
    splitted_seeds = {
        "train": [i for i in range(train_total)],
        "test": [i for i in range(train_total, train_total + test_total)],
    }
    jobs = []
    # print(M)
    start = time.time()
    for mode, mode_seeds in splitted_seeds.items():
        combinations = []

        prev_split = 0
        main_seed_count = 0
        # sort inverse by percents
        intersection_types = level_config[mode]["intersection_types"]
        intersections = sorted(
            [
                [_type, intersection_types[_type]["percent"]]
                for _type in intersection_types
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        log_dict[mode] = {x: {"count": 0, "percent": 0} for x in intersection_types}
        for intersection_type, intersection_percent in intersections:
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
                name_additions = [mode, level_name, intersection_type, speed]

                if level_name != "no-traffic":
                    name_additions.append(traffic_density)

                route_distributions = get_pattern(traffic_density, intersection_type)
                temp_seeds = seeds[inner_prev_split:inner_cur_split]
                seed_count += len(temp_seeds)
                if save_dir is None:
                    temp_save_dir = task_dir + "/" + "_".join(name_additions)
                else:
                    temp_save_dir = save_dir

                sub_proc = Process(
                    target=scenario_worker,
                    args=(
                        temp_seeds,
                        ego_mission,
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
                        dynamic_pattern_func,
                    ),
                )
                jobs.append(sub_proc)
                sub_proc.start()
                inner_prev_split = inner_cur_split
            print(
                f">> {mode} {intersection_type} count:{seed_count} generated: {seed_count/len(mode_seeds)} real: {intersection_percent}"
            )
            # print("--")
            prev_split = cur_split
            main_seed_count += seed_count
        # print(f"Finished: {mode}  {main_seed_count/(train_total+test_total)}")
        # print("--------------------------------------------")
    for process in jobs:
        process.join()
    print("*** time took:", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate scenarios")
    parser.add_argument("--task", help="type a task id [0, 1, 2, 3]", type=str)
    parser.add_argument("--level", help="easy/medium/hard, lo-hi/hi-lo", type=str)
    parser.add_argument(
        "--stopwatcher",
        help="all/aggressive/default/slow/blocker/crusher south-west",
        nargs="+",
    )
    parser.add_argument(
        "--save-dir", help="directory for saving maps", type=str, default=None
    )
    parser.add_argument(
        "--root-dir", help="directory of maps", type=str, default="ultra/scenarios"
    )

    args = parser.parse_args()

    stopwatcher_behavior, stopwatcher_route = None, None
    if args.stopwatcher:
        stopwatcher_behavior, stopwatcher_route = args.stopwatcher

    print("starting ...")
    build_scenarios(
        task=f"task{args.task}",
        level_name=args.level,
        stopwatcher_behavior=stopwatcher_behavior,
        stopwatcher_route=stopwatcher_route,
        save_dir=args.save_dir,
        root_path=args.root_dir,
    )
