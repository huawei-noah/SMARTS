import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

from zoo.evaluation.metrics.utils import map_agent_to_json_file


def evaluation_data_visualize(scenario_path, result_path, agent_list, agent_groups):
    # todo: get json file in scenario_path
    scenario_name = os.path.basename(scenario_path)
    json_data = []
    all_json_file = list(Path(scenario_path).glob("**/*json"))
    json_file_dict = map_agent_to_json_file(agent_list, all_json_file)
    for actor_name, agent_names in agent_groups.items():
        for agent_name in agent_names:
            assert (
                agent_name in json_file_dict
            ), f"{agent_name} has no entry in {json_file_dict}"

            with json_file_dict[agent_name].open() as f:
                json_result = json.load(f)
                json_data.append(json_result)

        agent_num = len(agent_names)
        fig, ax = plt.subplots(nrows=agent_num, ncols=2, figsize=(14, 14))
        fig.subplots_adjust(hspace=0.8)
        axes = ax.flatten()
        names = {}
        fig.suptitle(scenario_name, fontsize=15)
        for i in range(agent_num):
            names["ax_path" + str(i)], names["ax_speed" + str(i)] = (
                axes[2 * i + 0],
                axes[2 * i + 1],
            )
            plt.subplots_adjust(hspace=0.3)
            names["ax_path" + str(i)].tick_params(direction="in", top=True, right=True)
            agent_name = agent_names[i]
            names["ax_path" + str(i)].set_title(
                "Path_Curve-%s" % agent_name, fontsize=10
            )
            names["ax_path" + str(i)].set_xlabel("x/m")
            names["ax_path" + str(i)].set_ylabel("y/m")
            names["ax_speed" + str(i)].set_title(
                "Speed_Curve-%s" % agent_name, fontsize=10
            )
            names["ax_speed" + str(i)].set_xlabel("time/s")
            names["ax_speed" + str(i)].set_ylabel("speed/m/s")

            agent1_run_time = json_data[i]["agent"]["time_list"]
            agent1_run_speed = [
                j - j % 0.1 for j in json_data[i]["agent"]["speed_list"]
            ]
            agent1_pos_x = [j[0] for j in json_data[i]["agent"]["cartesian_pos_list"]]
            agent1_pos_y = [j[1] for j in json_data[i]["agent"]["cartesian_pos_list"]]

            npc1_run_time, npc1_run_speed, npc1_pos_x, npc1_pos_y = [], [], [], []
            if json_data[i]["npc"]:
                npc_data = json_data[i]["npc"][0]
                npc1_run_time = npc_data["time_list"]
                npc1_run_speed = [j - j % 0.1 for j in npc_data["speed_list"]]
                npc1_pos_x = [j[0] for j in npc_data["cartesian_pos_list"]]
                npc1_pos_y = [j[1] for j in npc_data["cartesian_pos_list"]]

            if npc1_run_time:
                valmax_pos_x = max(max(agent1_pos_x), max(npc1_pos_x))
                valmax_pos_y = max(max(agent1_pos_y), max(npc1_pos_y))
                valmin_pos_x = min(min(agent1_pos_x), min(npc1_pos_x))
                valmin_pos_y = min(min(agent1_pos_y), min(npc1_pos_y))
                valgap_pos_x = valmax_pos_x - valmin_pos_x
                valgap_pos_y = valmax_pos_y - valmin_pos_y
                valmax_speed = max(max(agent1_run_speed), max(npc1_run_speed))
                valmax_time = max(max(agent1_run_time), max(npc1_run_time))
                valmin_speed = min(min(agent1_run_speed), min(npc1_run_speed))
                valmin_time = min(min(agent1_run_time), min(npc1_run_time))
                valgap_speed = valmax_speed - valmin_speed
                valgap_time = valmax_time - valmin_time
            else:
                valmax_pos_x = max(agent1_pos_x)
                valmax_pos_y = max(agent1_pos_y)
                valmin_pos_x = min(agent1_pos_x)
                valmin_pos_y = min(agent1_pos_y)
                valgap_pos_x = valmax_pos_x - valmin_pos_x
                valgap_pos_y = valmax_pos_y - valmin_pos_y
                valmax_speed = max(agent1_run_speed)
                valmax_time = max(agent1_run_time)
                valmin_speed = min(agent1_run_speed)
                valmin_time = min(agent1_run_time)
                valgap_speed = valmax_speed - valmin_speed
                valgap_time = valmax_time - valmin_time

            if valgap_pos_y:
                names["ax_path" + str(i)].set_ylim(
                    [
                        (valmin_pos_y - 0.2 * valgap_pos_y),
                        (valmax_pos_y + 0.25 * valgap_pos_y),
                    ]
                )
            if valgap_speed:
                names["ax_speed" + str(i)].set_ylim(
                    [
                        (valmin_speed - 0.2 * valgap_speed),
                        (valmax_speed + 0.25 * valgap_speed),
                    ]
                )
            names["ax_path" + str(i)].set_xlim(
                [
                    (valmin_pos_x - 0.15 * valgap_pos_x),
                    (valmax_pos_x + 0.15 * valgap_pos_x),
                ]
            )
            names["ax_speed" + str(i)].set_xlim(
                [(valmin_time - 0.15 * valgap_time), (valmax_time + 0.15 * valgap_time)]
            )

            names["ax_path" + str(i)].plot(
                agent1_pos_x, agent1_pos_y, ":g", label="ego"
            )
            names["ax_speed" + str(i)].plot(
                agent1_run_time, agent1_run_speed, ":g", label="ego"
            )
            names["ax_path" + str(i)].legend(loc=0, ncol=2)
            names["ax_speed" + str(i)].legend(loc=0, ncol=2)

            vector_x1 = (
                agent1_pos_x[int(len(agent1_pos_x) / 2) + 1]
                - agent1_pos_x[int(len(agent1_pos_x) / 2)]
            )
            vector_y1 = (
                agent1_pos_y[int(len(agent1_pos_y) / 2) + 1]
                - agent1_pos_y[int(len(agent1_pos_y) / 2)]
            )
            if valgap_pos_y:
                names["ax_path" + str(i)].quiver(
                    agent1_pos_x[int(len(agent1_pos_x) / 2)],
                    agent1_pos_y[int(len(agent1_pos_y) / 2)],
                    vector_x1 / (valgap_pos_x / valgap_pos_y),
                    vector_y1,
                    width=0.005,
                    headwidth=5,
                    color="g",
                )
            else:
                names["ax_path" + str(i)].quiver(
                    agent1_pos_x[int(len(agent1_pos_x) / 2)],
                    agent1_pos_y[int(len(agent1_pos_y) / 2)],
                    vector_x1,
                    vector_y1,
                    width=0.005,
                    headwidth=5,
                    color="g",
                )
            names["ax_path" + str(i)].text(
                agent1_pos_x[0],
                agent1_pos_y[0],
                "time:" + str(agent1_run_time[0]),
                fontsize=6,
                color="g",
                style="normal",
            )
            names["ax_path" + str(i)].text(
                agent1_pos_x[int(len(agent1_pos_x) / 2)],
                agent1_pos_y[int(len(agent1_pos_y) / 2)],
                "time:" + str(agent1_run_time[int(len(agent1_run_time) / 2)]),
                fontsize=6,
                color="g",
                style="normal",
            )
            names["ax_path" + str(i)].text(
                agent1_pos_x[-1],
                agent1_pos_y[-1],
                "time:" + str(agent1_run_time[-1]),
                fontsize=6,
                color="g",
                style="normal",
            )

            if npc1_run_time:
                names["ax_path" + str(i)].plot(
                    npc1_pos_x, npc1_pos_y, ":r", label="npc"
                )
                names["ax_speed" + str(i)].plot(
                    npc1_run_time, npc1_run_speed, ":r", label="npc"
                )
                names["ax_path" + str(i)].legend(loc=0, ncol=2)
                names["ax_speed" + str(i)].legend(loc=0, ncol=2)
                names["ax_path" + str(i)].text(
                    npc1_pos_x[0],
                    npc1_pos_y[0],
                    "time:" + str(npc1_run_time[0]),
                    fontsize=6,
                    color="r",
                    style="normal",
                )
                names["ax_path" + str(i)].text(
                    npc1_pos_x[int(len(npc1_pos_x) / 2)],
                    npc1_pos_y[int(len(npc1_pos_y) / 2)],
                    "time:" + str(npc1_run_time[int(len(npc1_run_time) / 2)]),
                    fontsize=6,
                    color="r",
                    style="normal",
                )
                names["ax_path" + str(i)].text(
                    npc1_pos_x[-1],
                    npc1_pos_y[-1],
                    "time:" + str(npc1_run_time[-1]),
                    fontsize=6,
                    color="r",
                    style="normal",
                )

                vector_x2 = (
                    npc1_pos_x[int(len(npc1_pos_x) / 2) + 1]
                    - npc1_pos_x[int(len(npc1_pos_x) / 2)]
                )
                vector_y2 = (
                    npc1_pos_y[int(len(npc1_pos_y) / 2) + 1]
                    - npc1_pos_y[int(len(npc1_pos_y) / 2)]
                )
                if valgap_pos_y:
                    names["ax_path" + str(i)].quiver(
                        npc1_pos_x[int(len(npc1_pos_x) / 2)],
                        npc1_pos_y[int(len(npc1_pos_y) / 2)],
                        vector_x2 / (valgap_pos_x / valgap_pos_y),
                        vector_y2,
                        width=0.005,
                        headwidth=5,
                        color="r",
                    )
                else:
                    names["ax_path" + str(i)].quiver(
                        npc1_pos_x[int(len(npc1_pos_x) / 2)],
                        npc1_pos_y[int(len(npc1_pos_y) / 2)],
                        vector_x2,
                        vector_y2,
                        width=0.005,
                        headwidth=5,
                        color="r",
                    )
        time_suffix = datetime.now().strftime("%Y%m%d-%H%M")
        scenario_name_path = os.path.join(result_path + "/" + scenario_name)
        if not os.path.exists(scenario_name_path):
            os.mkdir(scenario_name_path)
        result_json_path = os.path.join(scenario_name_path + "/" + actor_name)
        os.mkdir(os.path.join(scenario_name_path + "/" + actor_name))
        result_file = os.path.join(
            result_json_path,
            "evaluation-curve_%s_%s.png" % (scenario_name, time_suffix),
        )
        plt.savefig(result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualization",
        description="Start run visualization.",
    )
    parser.add_argument(
        "log_path", help="The path to the run all evaluation origin data", type=str
    )
    parser.add_argument(
        "result_path", help="The path to the run all evaluation results", type=str
    )
    parser.add_argument("agent_list", help="All agent name list", type=str)
    parser.add_argument("agent_groups", help="Agent groups", type=str)
    args = parser.parse_args()
    evaluation_data_visualize(
        args.log_path, args.result_path, eval(args.agent_list), eval(args.agent_groups)
    )
