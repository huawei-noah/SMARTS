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
import os, sys, argparse

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc
import numpy as np
import matplotlib.pyplot as plt
from ultra.scenarios.generate_scenarios import build_scenarios
import time

# sumo -c  --remote-port=8813
# sumo -n map.net.xml -r traffic/all.rou.xml --remote-port=8813

# Aggregates information for each edge and lane ...
def edge_lane_data_function(traci, data_aggregator, **kwargs):
    if "lane_data" not in data_aggregator:
        data_aggregator["lane_data"] = {}
        for lane_id in traci.lane.getIDList():
            data_aggregator["lane_data"][lane_id] = {
                "num_vehicles": [],
                "occupancy": [],
            }
    if "edge_data" not in data_aggregator:
        data_aggregator["edge_data"] = {}
        for edge_id in traci.edge.getIDList():
            data_aggregator["edge_data"][edge_id] = {
                "num_vehicles": [],
                "occupancy": [],
            }

    for lane_id in traci.lane.getIDList():
        data_aggregator["lane_data"][lane_id]["num_vehicles"].append(
            traci.lane.getLastStepVehicleNumber(lane_id)
        )
        data_aggregator["lane_data"][lane_id]["occupancy"].append(
            traci.lane.getLastStepOccupancy(lane_id)
        )
    for edge_id in traci.edge.getIDList():
        data_aggregator["edge_data"][edge_id]["num_vehicles"].append(
            traci.edge.getLastStepVehicleNumber(edge_id)
        )
        data_aggregator["edge_data"][edge_id]["occupancy"].append(
            traci.edge.getLastStepOccupancy(edge_id)
        )


def vehicle_data_function(traci, data_aggregator, blocked_max_speed=0.1, **kwargs):
    vehicle_list = traci.vehicle.getIDList()
    if "vehicle_data" not in data_aggregator:
        data_aggregator["vehicle_data"] = {}

    if "destination" not in data_aggregator:
        data_aggregator["destination"] = {}

    if "route" not in data_aggregator:
        data_aggregator["route"] = {}

    if "vehicle_density" not in data_aggregator:
        data_aggregator["vehicle_density"] = {"All": {}}

    for veh_id in vehicle_list:
        if veh_id not in data_aggregator["vehicle_data"]:
            veh_route = traci.vehicle.getRoute(veh_id)
            data_aggregator["vehicle_density"]["All"][kwargs["last_step"]] = len(
                vehicle_list
            )
            data_aggregator["vehicle_data"][veh_id] = {
                "min_gap": traci.vehicle.getMinGap(veh_id),
                "final_target": veh_route[-1],
                "origin": veh_route[0],
                "destination": veh_route[-1],
                "success": False,
                "route": veh_route[0] + "|" + veh_route[-1],
                "rerouted": None,
                "travel_time": 0,
                "blocked_time": 0,
                "accumulated_blocked_time": 0,
                "positions": [],
                "lane_list": [],
                "last_step": kwargs["last_step"],
            }
            if veh_route[-1] not in data_aggregator["destination"]:
                data_aggregator["destination"][veh_route[-1]] = 1
            else:
                data_aggregator["destination"][veh_route[-1]] += 1

            if veh_route[0] + "|" + veh_route[-1] not in data_aggregator["route"]:
                data_aggregator["route"][veh_route[0] + "|" + veh_route[-1]] = 1
            else:
                data_aggregator["route"][veh_route[0] + "|" + veh_route[-1]] += 1
        else:
            data_aggregator["vehicle_data"][veh_id]["travel_time"] += 1
            data_aggregator["vehicle_data"][veh_id]["positions"].append(
                traci.vehicle.getPosition(veh_id)
            )

            if traci.vehicle.getSpeed(veh_id) < blocked_max_speed:
                data_aggregator["vehicle_data"][veh_id]["blocked_time"] += 1
                if traci.vehicle.getLeader(veh_id) is None:
                    data_aggregator["vehicle_data"][veh_id][
                        "accumulated_blocked_time"
                    ] += 1
            else:
                data_aggregator["vehicle_data"][veh_id]["accumulated_blocked_time"] = 0

        if (
            data_aggregator["vehicle_data"][veh_id]["route"]
            not in data_aggregator["vehicle_density"]
        ):
            data_aggregator["vehicle_density"][
                data_aggregator["vehicle_data"][veh_id]["route"]
            ] = {}
        else:
            if (
                not kwargs["last_step"]
                in data_aggregator["vehicle_density"][
                    data_aggregator["vehicle_data"][veh_id]["route"]
                ]
            ):
                data_aggregator["vehicle_density"][
                    data_aggregator["vehicle_data"][veh_id]["route"]
                ][kwargs["last_step"]] = 0
            else:
                data_aggregator["vehicle_density"][
                    data_aggregator["vehicle_data"][veh_id]["route"]
                ][kwargs["last_step"]] += 1

        current_lane = traci.vehicle.getLaneID(veh_id)
        if not current_lane in data_aggregator["vehicle_data"][veh_id]["lane_list"]:
            data_aggregator["vehicle_data"][veh_id]["lane_list"].append(current_lane)
        data_aggregator["vehicle_data"][veh_id][
            "current_edge"
        ] = traci.vehicle.getRoadID(veh_id)
        if (
            data_aggregator["vehicle_data"][veh_id]["current_edge"]
            == data_aggregator["vehicle_data"][veh_id]["final_target"]
        ):
            data_aggregator["vehicle_data"][veh_id]["success"] = True


def sumo_rerouting_routine(
    traci, data_aggregator, time_elapsed_until_reroute=20, **kwargs
):
    vehicle_list = traci.vehicle.getIDList()
    for veh_id in vehicle_list:
        vehicle_lane_index = traci.vehicle.getLaneIndex(veh_id)
        bestLanes = traci.vehicle.getBestLanes(veh_id)
        if not bestLanes[vehicle_lane_index][4] and len(bestLanes) > 1:

            selected_lanes = []
            for j in range(len(bestLanes)):
                if j != vehicle_lane_index:
                    selected_lanes.append(j)

            select_lane = np.random.choice(selected_lanes)
            if select_lane > vehicle_lane_index:
                direction = 1
            else:
                direction = -1

            # Is changing lane, locked, and is leader of the lane -> Reroute
            if (
                not traci.vehicle.couldChangeLane(veh_id, direction)
                and data_aggregator["vehicle_data"][veh_id]["accumulated_blocked_time"]
                > time_elapsed_until_reroute
                and traci.vehicle.getLeader(veh_id) is None
            ):
                origin = data_aggregator["vehicle_data"][veh_id]["origin"].split("-")[1]
                final_target = data_aggregator["vehicle_data"][veh_id][
                    "final_target"
                ].split("-")[1]
                # print(origin, final_target)
                potential_list = []
                for potential_reroute_target in data_aggregator["destination"]:
                    token = potential_reroute_target.split("-")[1]
                    if token != final_target and token != origin:
                        potential_list.append(potential_reroute_target)
                result = np.random.choice(potential_list)
                try:
                    # traci.vehicle.changeTarget(veh_id, result)
                    new_route = list(
                        traci.simulation.findRoute(
                            data_aggregator["vehicle_data"][veh_id]["current_edge"],
                            result,
                        ).edges
                    )
                    traci.vehicle.setRoute(veh_id, new_route)
                    print("Route Change Success")
                    data_aggregator["vehicle_data"][veh_id][
                        "accumulated_blocked_time"
                    ] = 0
                    data_aggregator["vehicle_data"][veh_id]["rerouted"] = result
                except:
                    print("Route Change Failed")


def generate_figures(
    data,
    figure_dir,
    setting_id,
    failure_rate,
    num_steps,
    num_bins=50,
    comparisons=[[("route", "avg_blocked_time"), ("route", "failure_rate")]],
):
    # default_statistics = {"blocked_time":[], "travel_time":[], "success":0, "rerouted":0, "in_simulation":0}
    total_travel_time_list = []
    travel_percentile_list = []
    blocked_travel_time_list = []
    num_vehicle_list = []
    statistics_list = []
    failed_list = []
    num_scenarios = len(data)
    num_bins = min(num_scenarios // 2, num_bins)
    for scenario in data:
        statistics = {
            "destination": {},
            "route": {},
            "vehicle_gen_prob": scenario["vehicle_gen_prob"],
        }
        scenario_travel_time = []
        scenario_blocked_time = []
        for veh_id in scenario["vehicle_data"]:
            destination_key = scenario["vehicle_data"][veh_id]["destination"]
            route_key = scenario["vehicle_data"][veh_id]["route"]

            is_successful = scenario["vehicle_data"][veh_id]["success"]
            is_rerouted = not scenario["vehicle_data"][veh_id]["rerouted"] is None
            is_still_running = not is_successful and not is_rerouted
            travel_time = scenario["vehicle_data"][veh_id]["travel_time"]
            blocked_time = scenario["vehicle_data"][veh_id]["blocked_time"]
            has_failed = travel_time > failure_rate or is_rerouted

            if not destination_key in statistics["destination"]:
                statistics["destination"][destination_key] = {
                    "blocked_time": [],
                    "travel_time": [],
                    "success": 0,
                    "rerouted": 0,
                    "in_simulation": 0,
                    "deadlock_rate": 0,
                    "num_vehicles": 0,
                }
            if not route_key in statistics["route"]:
                statistics["route"][route_key] = {
                    "blocked_time": [],
                    "travel_time": [],
                    "success": 0,
                    "rerouted": 0,
                    "in_simulation": 0,
                    "deadlock_rate": 0,
                    "num_vehicles": 0,
                }

            statistics["destination"][destination_key]["success"] += int(is_successful)
            statistics["destination"][destination_key]["rerouted"] += int(is_rerouted)
            statistics["destination"][destination_key]["in_simulation"] += int(
                is_still_running
            )
            statistics["destination"][destination_key]["deadlock_rate"] += int(
                has_failed
            )
            statistics["destination"][destination_key]["num_vehicles"] += 1

            statistics["destination"][destination_key]["travel_time"].append(
                travel_time
            )
            statistics["destination"][destination_key]["blocked_time"].append(
                blocked_time
            )

            statistics["route"][route_key]["success"] += int(is_successful)
            statistics["route"][route_key]["rerouted"] += int(is_rerouted)
            statistics["route"][route_key]["in_simulation"] += int(is_still_running)
            statistics["route"][route_key]["deadlock_rate"] += int(has_failed)
            statistics["route"][route_key]["num_vehicles"] += 1

            statistics["route"][route_key]["travel_time"].append(travel_time)
            statistics["route"][route_key]["blocked_time"].append(blocked_time)

        travel_time_list = []
        blocked_time_list = []
        scenario_failed = 0
        for destination_key in statistics["destination"]:
            travel_time_list.extend(
                statistics["destination"][destination_key]["travel_time"]
            )
            blocked_time_list.extend(
                statistics["destination"][destination_key]["blocked_time"]
            )
            scenario_failed += statistics["destination"][destination_key][
                "deadlock_rate"
            ]

        scenario_99_percentile_travel_time = np.percentile(travel_time_list, 99)
        scenario_total_travel_time = np.mean(travel_time_list)
        scenario_blocked_time = np.mean(blocked_time_list)

        total_travel_time_list.append(scenario_total_travel_time)
        travel_percentile_list.append(scenario_99_percentile_travel_time)
        blocked_travel_time_list.append(scenario_blocked_time)

        failed_list.append(scenario_failed / max(len(scenario["vehicle_data"]), 1))
        num_vehicle_list.append(len(scenario["vehicle_data"]))
        statistics_list.append(statistics)

    index = 0
    for scenario in data:
        statistics = statistics_list[index]
        scenario_statistics = {}
        route_list = list(scenario["vehicle_density"].keys())
        for route_id in route_list:
            scenario_statistics[route_id] = []
        for i in range(num_steps):
            for route_id in route_list:
                if not i in scenario["vehicle_density"][route_id]:
                    scenario["vehicle_density"][route_id][i] = 0
                scenario_statistics[route_id].append(
                    scenario["vehicle_density"][route_id][i]
                )
        for route_id in route_list:
            scenario_statistics[route_id] = np.mean(scenario_statistics[route_id])
        statistics["vehicle_density"] = scenario_statistics
        index += 1

    # Append all scenario statistics together
    all_statistics = {
        "destination": {},
        "route": {},
        "vehicle_density": {},
        "vehicle_gen_prob": {},
    }
    all_statistics["route"]["All"] = {
        "avg_blocked_time": blocked_travel_time_list,
        "avg_travel_time": total_travel_time_list,
        "deadlock_rate": failed_list,
        "num_vehicles": num_vehicle_list,
    }

    for scenario_statistics in statistics_list:
        for destination_key in scenario_statistics["destination"]:
            if not destination_key in all_statistics["destination"]:
                all_statistics["destination"][destination_key] = {
                    "blocked_time": [],
                    "travel_time": [],
                    "success": [],
                    "rerouted": [],
                    "in_simulation": [],
                    "deadlock_rate": [],
                    "num_vehicles": [],
                }
            all_statistics["destination"][destination_key]["success"].append(
                scenario_statistics["destination"][destination_key]["success"]
            )
            all_statistics["destination"][destination_key]["rerouted"].append(
                scenario_statistics["destination"][destination_key]["rerouted"]
            )
            all_statistics["destination"][destination_key]["in_simulation"].append(
                scenario_statistics["destination"][destination_key]["in_simulation"]
            )
            all_statistics["destination"][destination_key]["travel_time"].extend(
                scenario_statistics["destination"][destination_key]["travel_time"]
            )
            all_statistics["destination"][destination_key]["blocked_time"].extend(
                scenario_statistics["destination"][destination_key]["blocked_time"]
            )

            all_statistics["destination"][destination_key]["deadlock_rate"].append(
                scenario_statistics["destination"][destination_key]["deadlock_rate"]
                / max(
                    scenario_statistics["destination"][destination_key]["num_vehicles"],
                    1,
                )
            )
            all_statistics["destination"][destination_key]["num_vehicles"].append(
                scenario_statistics["destination"][destination_key]["num_vehicles"]
            )

        for route_key in scenario_statistics["route"]:
            if not route_key in all_statistics["route"]:
                all_statistics["route"][route_key] = {
                    "blocked_time": [],
                    "travel_time": [],
                    "avg_travel_time": [],
                    "avg_blocked_time": [],
                    "success": [],
                    "rerouted": [],
                    "in_simulation": [],
                    "deadlock_rate": [],
                    "num_vehicles": [],
                }
            all_statistics["route"][route_key]["success"].append(
                scenario_statistics["route"][route_key]["success"]
            )
            all_statistics["route"][route_key]["rerouted"].append(
                scenario_statistics["route"][route_key]["rerouted"]
            )
            all_statistics["route"][route_key]["in_simulation"].append(
                scenario_statistics["route"][route_key]["in_simulation"]
            )
            all_statistics["route"][route_key]["travel_time"].extend(
                scenario_statistics["route"][route_key]["travel_time"]
            )
            all_statistics["route"][route_key]["blocked_time"].extend(
                scenario_statistics["route"][route_key]["blocked_time"]
            )

            all_statistics["route"][route_key]["avg_travel_time"].append(
                np.mean(scenario_statistics["route"][route_key]["travel_time"])
            )
            all_statistics["route"][route_key]["avg_blocked_time"].append(
                np.mean(scenario_statistics["route"][route_key]["blocked_time"])
            )

            all_statistics["route"][route_key]["deadlock_rate"].append(
                scenario_statistics["route"][route_key]["deadlock_rate"]
                / max(scenario_statistics["route"][route_key]["num_vehicles"], 1)
            )
            all_statistics["route"][route_key]["num_vehicles"].append(
                scenario_statistics["route"][route_key]["num_vehicles"]
            )
        for route_key in scenario_statistics["vehicle_density"]:
            if route_key not in all_statistics["vehicle_density"]:
                all_statistics["vehicle_density"][route_key] = {"vehicle_density": []}
            all_statistics["vehicle_density"][route_key]["vehicle_density"].append(
                scenario_statistics["vehicle_density"][route_key]
            )

        for route_key in scenario_statistics["vehicle_gen_prob"]:
            if route_key not in all_statistics["vehicle_gen_prob"]:
                all_statistics["vehicle_gen_prob"][route_key] = {"vehicle_gen_prob": []}
            all_statistics["vehicle_gen_prob"][route_key]["vehicle_gen_prob"].append(
                scenario_statistics["vehicle_gen_prob"][route_key]
            )
        # For each destination and route, we generate 5 histogram ....  + 2 for the travel time and percentile ...
    try:
        os.mkdir(figure_dir)
    except:
        print("Directories already exists ... ")
    figure_dir = figure_dir + "/" + setting_id
    try:
        os.mkdir(figure_dir)
    except:
        print("Directories already exists ... ")

    for overarching_key in all_statistics:
        for data_key in all_statistics[overarching_key]:
            sub_figure_dir = figure_dir + "/" + data_key
            try:
                os.mkdir(sub_figure_dir)
            except:
                print("Directories already exists ... ")
            for statistics_key in all_statistics[overarching_key][data_key]:
                n, bins, patches = plt.hist(
                    all_statistics[overarching_key][data_key][statistics_key],
                    num_bins,
                    facecolor="blue",
                )
                plt.title(
                    "Histogram Edge/Route_" + str(data_key) + "_" + str(statistics_key)
                )
                plt.xlabel(statistics_key)
                plt.ylabel(
                    "Number of occurences for " + str(num_scenarios) + " scenarios"
                )
                plt.savefig(sub_figure_dir + "/" + statistics_key + ".png")
                plt.close()

    # collect key informations
    for i in range(len(comparisons)):
        comparison = comparisons[i]
        key_00 = comparison[0][0]
        key_10 = comparison[1][0]
        key_01 = comparison[0][1]
        key_11 = comparison[1][1]
        print(comparison)
        id_list = list(all_statistics[key_00])
        for key_id in id_list:
            set_1 = all_statistics[key_00][key_id][key_01]
            set_2 = all_statistics[key_10][key_id][key_11]
            if len(set_1) == len(set_2):
                plt.plot(set_1, set_2, ".")
                plt.title(key_01 + " vs " + key_11)
                plt.xlabel(key_01)
                plt.ylabel(key_11)
                plt.savefig(figure_dir + "/" + key_id + "/" + key_01 + " vs " + key_11)
                plt.close()
            else:
                print("Invalid Comparison")


def sumo_traci_runner(
    simulation_time,
    custom_traci_functions,
    num_iterations,
    scenario_dir,
    simulation_step_size=1,
    init_time_skip=50,
    seed=0,
    run_per_scenario=2,
    **kwargs,
):
    skip = int(1 / simulation_step_size)
    np.random.seed(seed)
    all_simulation_data = []
    traci.init(8813)
    task = kwargs["task"]
    level = kwargs["level"]
    print(task, level)

    def dynamic_vehicle_gen_probability_func(pattern, current_step):
        if kwargs["vehicle_gen_prob"] > 0:
            current_step = current_step // run_per_scenario
            for edge_id in pattern["routes"]:
                if pattern["routes"][edge_id] is None:
                    continue
                pattern["routes"][edge_id]["begin_time_init"]["params"][
                    "probability"
                ] = (
                    kwargs["vehicle_gen_prob"]
                    + current_step * kwargs["dynamic_vehicle_gen_prob"]
                )
        return pattern

    build_scenarios(
        task=f"task{task}",
        level_name=level,
        num_seeds=num_iterations * run_per_scenario,
        has_stopwatcher=False,
        dynamic_pattern_func=dynamic_vehicle_gen_probability_func,
    )
    for j in range(num_iterations * run_per_scenario):
        current_time = time.time()
        current_scenario_dir = scenario_dir + "-flow-" + str(j)
        current_vehicle_gen_prob = (
            kwargs["vehicle_gen_prob"]
            + j // run_per_scenario * kwargs["dynamic_vehicle_gen_prob"]
        )
        # for k in range(run_per_scenario):
        #     # Reset scenario
        traci.load(
            [
                "-n",
                current_scenario_dir + "/map.net.xml",
                "-r",
                current_scenario_dir + "/traffic/all.rou.xml",
                "--seed",
                str(np.random.randint(0, run_per_scenario)),
            ]
        )
        simulation_data = {}  # Dictionary object to collect data ...
        accumulated_time = 0
        for i in range(int((simulation_time + init_time_skip) / simulation_step_size)):
            accumulated_time += simulation_step_size
            traci.simulationStep(accumulated_time)  # One step to initialize everything
            if i > int(init_time_skip / simulation_step_size) and i % skip == 0:
                # print("Sim:", j, "Sim Seed:", k,"Time Step:", traci.simulation.getTime(), "Num Vehicles:", len(traci.vehicle.getIDList()))
                for func in custom_traci_functions:
                    func(
                        traci,
                        simulation_data,
                        last_step=i - init_time_skip - 1,
                        **kwargs,
                    )

        simulation_data["vehicle_gen_prob"] = {"All": current_vehicle_gen_prob}
        for route_id in simulation_data["route"]:
            simulation_data["vehicle_gen_prob"][route_id] = current_vehicle_gen_prob

            all_simulation_data.append(simulation_data)
        print(
            "Sim:",
            j,
            "with",
            run_per_scenario,
            "seeds, completed with",
            time.time() - current_time,
            "seconds",
        )
    traci.close()
    # Reached End of simulation ...
    return all_simulation_data


# Default sumo instruction for 4 lane_t
# sumo -n research/edmonton/intersections/scenarios/task1/hard_4lane_t_80kmh_heavy_traffic_stress_test_t_intersection-mission-0-flow-0/map.net.xml -r research/edmonton/intersections/scenarios/task1/hard_4lane_t_80kmh_heavy_traffic_stress_test_t_intersection-mission-0-flow-0/traffic/all.rou.xml --remote-port=8813
# Run command example
# python research/edmonton/intersections/scenarios/sumo_experiment.py --level="hard" --task="1" --scenario_dir="research/edmonton/intersections/scenarios/task1/hard_4lane_t_80kmh_heavy_traffic_stress_test_t_intersection-mission-0" --figure_dir="research/edmonton/intersections/scenarios/task1/figures" --setting_id="hard_t_intersection_4_vehicle_prob_0_05" --num_scenario=100
if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate scenarios")
    parser.add_argument("--task", help="type a task id [0, 1, 2, 3]", type=str)
    parser.add_argument("--level", help="easy/medium/hard", type=str)
    parser.add_argument("--scenario_dir", help="path", type=str)
    parser.add_argument("--figure_dir", help="path", type=str)
    parser.add_argument(
        "--dynamic_vehicle_gen_prob", help="path", type=float, default=0
    )
    parser.add_argument("--vehicle_gen_prob", help="path", type=float, default=0.05)
    parser.add_argument("--run_per_scenario", help="path", type=int, default=3)

    parser.add_argument(
        "--setting_id",
        help="for identifying figures",
        type=str,
        default="hard_t_intersection_0_05",
    )
    parser.add_argument("--reroute", help="path", type=str, default="False")
    parser.add_argument("--num_scenario", help="path", type=int, default=5)
    parser.add_argument("--scenario_run_time", help="path", type=int, default=1200)
    parser.add_argument("--failure_time", help="path", type=int, default=600)

    args = parser.parse_args()
    if args.level not in [
        "easy",
        "medium",
        "hard",
    ]:
        raise ValueError('level not accepted select from ["easy", "medium", "hard"]')
    if args.reroute == "True":
        sim_funcs = [
            edge_lane_data_function,
            vehicle_data_function,
            sumo_rerouting_routine,
        ]
    else:
        sim_funcs = [edge_lane_data_function, vehicle_data_function]
    data = sumo_traci_runner(
        args.scenario_run_time,
        sim_funcs,
        args.num_scenario,
        args.scenario_dir,
        task=args.task,
        level=args.level,
        vehicle_gen_prob=args.vehicle_gen_prob,
        dynamic_vehicle_gen_prob=args.dynamic_vehicle_gen_prob,
        run_per_scenario=args.run_per_scenario,
    )
    comparisons = [
        [("route", "avg_blocked_time"), ("route", "deadlock_rate")],
        [("vehicle_density", "vehicle_density"), ("route", "deadlock_rate")],
        [("vehicle_density", "vehicle_density"), ("route", "avg_travel_time")],
        [("vehicle_density", "vehicle_density"), ("route", "avg_blocked_time")],
        [("vehicle_gen_prob", "vehicle_gen_prob"), ("route", "avg_blocked_time")],
        [("vehicle_gen_prob", "vehicle_gen_prob"), ("route", "deadlock_rate")],
        [
            ("vehicle_gen_prob", "vehicle_gen_prob"),
            ("vehicle_density", "vehicle_density"),
        ],
    ]
    generate_figures(
        data,
        args.figure_dir,
        args.setting_id
        + "_reroute_"
        + str(args.reroute)
        + "_failure_time_"
        + str(args.failure_time),
        args.failure_time,
        args.scenario_run_time,
        comparisons=comparisons,
    )
