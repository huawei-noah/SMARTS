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
import argparse, sys, os, glob, dill, ray, json, timeit
from collections import defaultdict
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    NeighborhoodVehicles,
)
from ultra.scenarios.generate_scenarios import build_scenarios
from ultra.scenarios.analysis.sumo_experiment import (
    edge_lane_data_function,
    vehicle_data_function,
    sumo_rerouting_routine,
)
from ultra.scenarios.analysis.scenario_analysis import ScenarioAnalysis
from ultra.scenarios.analysis.behavior_analysis import BehaviorAnalysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser("scenarios")
    subparsers = parser.add_subparsers(help="sub-command help")

    parser_generate_scenarios = subparsers.add_parser(
        "generate", help="Generating scenarios"
    )
    parser_generate_scenarios.add_argument(
        "--task", help="type a task id [0, 1, 2, 3]", type=str
    )
    parser_generate_scenarios.add_argument(
        "--save-dir", help="directory for saving maps", type=str, default=None
    )
    parser_generate_scenarios.add_argument(
        "--root-dir",
        help="directory for saving maps",
        type=str,
        default="ultra/scenarios",
    )
    parser_generate_scenarios.add_argument("--level", help="easy/medium/hard", type=str)
    parser_generate_scenarios.add_argument(
        "--stopwatcher",
        help="aggressive/default/slow/blocker/crusher south-west",
        nargs="+",
    )

    parser_generate_scenarios.set_defaults(which="generate")

    parser_analyze_scenarios = subparsers.add_parser(
        "analyze", help="Analyzing scenarios"
    )
    parser_analyze_scenarios.add_argument(
        "--scenarios", help="type a task id [0, 1, 2, 3]", type=str
    )
    parser_analyze_scenarios.add_argument("--output", help="easy/medium/hard", type=str)
    parser_analyze_scenarios.add_argument("--failure-time", type=int, default=600)
    parser_analyze_scenarios.add_argument("--max-steps", type=int, default=6000)
    parser_analyze_scenarios.add_argument(
        "--video",
        help="record video by rate (default every 100 scenarios)",
        type=int,
        default=1000,
    )
    parser_analyze_scenarios.add_argument("--ego", help="path to the trained agent")
    parser_analyze_scenarios.add_argument("--policy", help="ego agent policy")
    parser_analyze_scenarios.add_argument(
        "--end-by-stopwatcher",
        help="end episode as stopwatcher exits",
        action="store_true",
    )
    parser_analyze_scenarios.set_defaults(which="analyze")

    parser_profile_vehicles = subparsers.add_parser("behavior", help="Profile vehicles")
    parser_profile_vehicles.add_argument(
        "--scenarios", help="type a task id [0, 1, 2, 3]", type=str
    )
    parser_profile_vehicles.add_argument(
        "--video",
        help="record video by rate (default every 100 scenarios)",
        type=int,
        default=1000,
    )
    parser_profile_vehicles.add_argument(
        "--end-by-stopwatcher",
        help="end episode as stopwatcher exits",
        action="store_true",
    )
    parser_profile_vehicles.add_argument("--max-steps", type=int, default=6000)
    parser_profile_vehicles.add_argument("--output", help="easy/medium/hard", type=str)
    parser_profile_vehicles.set_defaults(which="behavior")

    parser_plot_data = subparsers.add_parser(
        "plot-analysis", help="Plot existing analysis data"
    )
    parser_plot_data.add_argument("--data", help="path to the load data", type=str)
    parser_plot_data.add_argument("--output", help="path to save data", type=str)
    parser_plot_data.add_argument("--failure_time", type=int, default=600)
    parser_plot_data.set_defaults(which="plot-analysis")

    args = parser.parse_args()
    if args.which == "generate":
        stopwatcher_behavior, stopwatcher_route = None, None
        if args.stopwatcher:
            stopwatcher_behavior, stopwatcher_route = args.stopwatcher
        build_scenarios(
            task=f"task{args.task}",
            level_name=args.level,
            stopwatcher_behavior=stopwatcher_behavior,
            stopwatcher_route=stopwatcher_route,
            save_dir=args.save_dir,
            root_path=args.root_dir,
        )
    else:
        ray.init()
        if args.which == "plot-analysis":
            analyzer = ScenarioAnalysis.remote()
            ray.wait([analyzer.load_data.remote(args.data)])
            ray.wait(
                [
                    analyzer.draw_plots.remote(
                        save_dir=args.output, failure_time=args.failure_time
                    )
                ]
            )
        else:
            start = timeit.default_timer()
            if args.which == "analyze":
                analyzer = ScenarioAnalysis.remote()
            elif args.which == "behavior":
                analyzer = BehaviorAnalysis.remote()
                args.ego, args.policy, args.failure_time = None, None, None

            scenarios = glob.glob(f"{args.scenarios}*")
            timestep_sec = 0.1
            os.makedirs(args.output, exist_ok=True)
            ray.wait(
                [
                    analyzer.run.remote(
                        end_by_stopwatcher=args.end_by_stopwatcher,
                        scenarios=scenarios,
                        ego=args.ego,
                        max_episode_steps=args.max_steps,
                        policy=args.policy,
                        video_rate=args.video,
                        timestep_sec=timestep_sec,
                        custom_traci_functions=[
                            edge_lane_data_function,
                            vehicle_data_function,
                        ],
                    )
                ]
            )
            end = timeit.default_timer()
            print("Time taken:", end - start)
            ray.wait([analyzer.save_data.remote(save_dir=args.output)])
            ray.wait(
                [
                    analyzer.draw_plots.remote(
                        save_dir=args.output, failure_time=args.failure_time
                    )
                ]
            )
