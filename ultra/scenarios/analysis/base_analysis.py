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
import numpy as np
import gym, random, re, timeit, copy
import glob, os, argparse, json, ray, torch
from matplotlib import pyplot as plt
import dill
from ast import literal_eval
from collections import defaultdict
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    NeighborhoodVehicles,
)
from ultra.scenarios.common.visualization import (
    draw_intersection,
    convert_to_gif,
    profile_vehicles,
)
from ultra.scenarios.common.social_vehicle_definitions import get_social_vehicle_color


class DefaultPolicy(Agent):
    def act(self, obs):
        lane_index = 0
        num_trajectory_points = min([10, len(obs.waypoint_paths[lane_index])])
        desired_speed = 20
        trajectory = [
            [
                obs.waypoint_paths[lane_index][i].pos[0]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].pos[1]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].heading
                for i in range(num_trajectory_points)
            ],
            [desired_speed for i in range(num_trajectory_points)],
        ]
        return trajectory


class BaseAnalysis:
    def __init__(self):
        self.social_vehicles_states = {}
        self.social_vehicles_ids = set()
        self.finished_social_vehicles_ids = set()
        self.analysis = {}

    def reset_scenario_cache(self):
        self.social_vehicles_ids = {}
        self.finished_social_vehicles_ids = set()
        self.social_vehicles_states = defaultdict(
            lambda: {
                "position": [0, 0],
                "route": (),
                "in_junction": [],
                "behavior": None,
                "steps": 0,
                "stop_step": 0,
                "start_step": None,
                "end_step": None,
                "edge": None,
                "speeds": [0],
                "accels": [0],
            }
        )

    def save_histogram(self, data, figure_name, title, x_range=None):
        plt.figure()

        n, bins, patches = plt.hist(
            x=data, bins=600, color="#0504aa", alpha=0.7, rwidth=0.85
        )
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel(title)
        plt.ylabel("Frequency")
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        if x_range is not None:
            plt.xlim(x_range)

        plt.savefig(figure_name)
        plt.close()

    def get_agent(self, ego, policy, max_episode_steps):
        observation_adapter = None
        if ego:
            config = get_agent_config_by_type(policy)
            agent_spec = AgentSpec(
                interface=config["interface"],
                policy_params=dict(config["policy"], checkpoint_dir=ego_model),
                policy_builder=config["policy_class"],
            )
            agent_spec.interface.max_episode_steps = max_episode_steps
            observation_adapter = IntersectionAdapter(
                agent_id="AGENT_007",
                social_vehicle_config=config["social_vehicle_config"],
                timestep_sec=config["env"]["timestep_sec"],
                **config["other"],
            )
        else:
            # Lane Following agent
            agent_spec = AgentSpec(
                interface=AgentInterface(
                    max_episode_steps=max_episode_steps,  # 10 mins
                    waypoints=True,
                    action=ActionSpaceType.Lane,
                    debug=False,
                    neighborhood_vehicles=NeighborhoodVehicles(radius=2000),
                ),
                agent_builder=DefaultPolicy,
            )
        return agent_spec, observation_adapter

    def process_social_vehicles(self, vehicles, dt, step):
        stopwatcher_state = None
        stopwatcher_exit = False
        current_vehicles = set()
        for v in vehicles:
            v_id = v.id
            is_new = False
            behavior_key, _ = get_social_vehicle_color(v.id)

            if v_id not in self.social_vehicles_ids:
                self.social_vehicles_ids[v_id] = len(self.social_vehicles_ids)
                is_new = True
            current_vehicles.add(self.social_vehicles_ids[v_id])
            # -----------------------------------
            # 1- Find route information
            # -----------------------------------
            _route = tuple(re.findall("edge-((....|.....)-..)", v_id))
            route = (_route[0][0], _route[1][0])

            previous_edge = self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                "edge"
            ]
            current_edge = re.findall("edge-(....|.....|junction*)-", v.edge_id)
            current_edge = current_edge[0] if len(current_edge) == 1 else None

            # -----------------------------------
            # 2- start caching
            # -----------------------------------
            if is_new:
                self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "start_step"
                ] = step
                self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "behavior"
                ] = behavior_key
                self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "route"
                ] = route

            # -----------------------------------
            # 3- apply updates
            # -----------------------------------
            # update position
            self.social_vehicles_states[self.social_vehicles_ids[v_id]]["position"] = [
                v.position[0],
                v.position[1],
            ]

            # stopped time
            if v.speed < 0.01:
                self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "stop_step"
                ] += 1

            # collect speed and accel:
            accel = (
                v.speed
                - self.social_vehicles_states[self.social_vehicles_ids[v_id]]["speeds"][
                    -1
                ]
            ) / dt
            self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                "speeds"
            ].append(v.speed)
            self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                "accels"
            ].append(accel)

            # in junction time range
            if "junction" in v.edge_id:
                n = len(
                    self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                        "speeds"
                    ]
                )
                if not self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "in_junction"
                ]:
                    self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                        "in_junction"
                    ] = [n, n]
                else:
                    self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                        "in_junction"
                    ][0] = min(
                        self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                            "in_junction"
                        ][0],
                        n,
                    )
                    self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                        "in_junction"
                    ][1] = max(
                        self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                            "in_junction"
                        ][1],
                        n,
                    )

            self.social_vehicles_states[self.social_vehicles_ids[v_id]]["steps"] += 1
            if current_edge:
                self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "edge"
                ] = current_edge

            # -------------------------------------
            # 3- end_time
            # -----------------------------------
            self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                "end_step"
            ] = step

            # check for stopwatcher vehicle
            if (
                "stopwatcher"
                in self.social_vehicles_states[self.social_vehicles_ids[v_id]][
                    "behavior"
                ]
            ):
                stopwatcher_state = self.social_vehicles_states[
                    self.social_vehicles_ids[v_id]
                ]

        # cleanup expired vehicles
        for v in list(self.social_vehicles_states.keys()):
            if v not in current_vehicles:
                self.finished_social_vehicles_ids.add(v)
                if "stopwatcher" in self.social_vehicles_states[v]["behavior"]:
                    stopwatcher_exit = True

        return stopwatcher_state, stopwatcher_exit

    # Number of GPUs should be splited between remote functions.
    def run(
        self,
        scenarios,
        timestep_sec,
        ego,
        policy,
        max_episode_steps,
        video_rate,
        custom_traci_functions,
        end_by_stopwatcher,
        analyze_func,
        init_time_skip=20,
    ):
        agent_spec, observation_adapter = self.get_agent(ego, policy, max_episode_steps)
        agent_id = "AGENT_007"
        ego_less = False if ego else True
        env = gym.make(
            "smarts.env:hiway-v0",
            scenarios=scenarios,
            agent_specs={agent_id: agent_spec},
            headless=True,
            visdom=False,
            timestep_sec=timestep_sec,
            sumo_headless=True,
            seed=0,
            endless_traffic=False,
        )
        visited_scenario = set()
        while len(visited_scenario) < len(scenarios):
            agent = agent_spec.build_agent()
            observations = env.reset()
            scenario_path = env._smarts.scenario.root_filepath

            # only run scenario once
            if scenario_path not in visited_scenario:
                with open(f"{scenario_path}/metadata.json", "r") as metadata_rd:
                    metadata = json.load(metadata_rd)
                visited_scenario.add(scenario_path)
                all_waypoints = [
                    [linked_wp.wp.pos[0], linked_wp.wp.pos[1]]
                    for linked_wp in env._smarts.waypoints._linked_waypoints
                ]
                dones = {"__all__": False}
                intersection_name, intersection_tag = (
                    metadata["intersection"]["name"],
                    metadata["intersection"]["type"],
                )
                total_vehicles = int(metadata["total_vehicles"])
                print(
                    f"Processing Scenario {len(visited_scenario)}/{len(scenarios)}:{scenario_path} "
                )
                print("Total number of vehicles", total_vehicles)
                print(f"Running egoless:{ego_less}")

                self.reset_scenario_cache()
                step, episode_time = 0, 0.0
                simulation_start = timeit.default_timer()
                stopwatcher_exit = False
                stopwatcher_logged = False
                stopwatcher_max_steps = 0
                images, simulation_data = [], {}
                while not dones["__all__"]:
                    if step % 200 == 0:
                        print("step", step)
                    # todo ------ add ego agent
                    agent_obs = observations[agent_id]
                    start = agent_obs.ego_vehicle_state.mission.start
                    goal = agent_obs.ego_vehicle_state.mission.goal
                    path = agent_obs.waypoint_paths[0]
                    agent_action = "slow_down"
                    observations, rewards, dones, infos = env.step(
                        {agent_id: agent_action}
                    )
                    # ------------
                    (
                        stopwatcher_state,
                        stopwatcher_exit,
                    ) = self.process_social_vehicles(
                        agent_obs.neighborhood_vehicle_states, timestep_sec, step,
                    )
                    if stopwatcher_state:
                        has_stopwatcher = True
                        stopwatcher_max_steps = max(
                            stopwatcher_max_steps, stopwatcher_state["steps"]
                        )
                        if not stopwatcher_logged:
                            print("stopwatcher detected!")
                            stopwatcher_logged = True
                    if (
                        step > (init_time_skip / timestep_sec)
                        and step % int(1 / timestep_sec) == 0
                    ):
                        for func in custom_traci_functions:
                            func(
                                env._smarts.traffic_sim._traci_conn,
                                simulation_data,
                                last_step=step,
                            )

                    step += 1
                    episode_time += timestep_sec

                    if len(visited_scenario) % video_rate == 0:
                        images.append(
                            draw_intersection(
                                ego=agent_obs.ego_vehicle_state,
                                social_vehicle_states=self.social_vehicles_states,
                                goal_path=path,
                                all_waypoints=all_waypoints,
                                step=step,
                                timestep_sec=timestep_sec,
                                goal=goal.position[0:2],
                                start=start.position[0:2],
                                intersection_tag=intersection_tag,
                                finished_vehicles=self.finished_social_vehicles_ids,
                            )
                        )
                    if ego_less and stopwatcher_exit and end_by_stopwatcher:
                        break
                    if len(self.finished_social_vehicles_ids) == total_vehicles:
                        break
                simulation_end = timeit.default_timer()
                if ego_less and stopwatcher_max_steps > 0:
                    print("Saved stopwatcher steps to", scenario_path)
                    with open(f"{scenario_path}/max_steps.txt", "w") as file_wr:
                        file_wr.write(str(stopwatcher_max_steps))
                simulation_time = simulation_end - simulation_start
                print(analyze_func)
                analyze_func(episode_time, stopwatcher_max_steps, simulation_data)
                print(
                    f"Episode finished in {simulation_time}s, {episode_time}s simulation"
                )

                if len(visited_scenario) % video_rate == 0:
                    print("Converting images to video...")
                    convert_to_gif(
                        images=images,
                        save_dir=scenario_path,
                        name=f"{intersection_name}-{metadata['flow_id']}",
                    )

                print("--------------------------------")

        env.close()
