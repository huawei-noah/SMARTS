# Add offline training code for Track-2 here.
# On completion of training, automatically save the trained model to `track2/submission` directory.

import argparse
import glob
import re
import numpy as np
import os
import pickle
import sys
import shutil
import yaml
from pathlib import Path
from PIL import Image
from typing import Any, Dict, Optional

import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.algos import CQL

# To import submission folder
sys.path.insert(0, str(Path(__file__).parents[1]))

from submission.utility import (
    goal_region_reward,
    get_goal_layer,
    get_trans_coor,
)


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def train(input_path, output_path):
    d3rlpy.seed(313)

    # Get config parameters.
    train_config = load_config(Path(__file__).absolute().parents[0] / "config.yaml")

    n_steps = train_config["n_steps"]
    n_steps_per_epoch = train_config["n_steps_per_epoch"]
    n_scenarios = train_config["n_scenarios"]
    n_vehicles = train_config["n_vehicles"]
    gpu = train_config["gpu"]

    scenarios = list()
    for scenario_name in os.listdir(input_path):
        scenarios.append(scenario_name)

    save_directory = Path(__file__).absolute().parents[0] / "d3rlpy_logs"
    if not os.path.isdir(save_directory):
        index = 0
        os.mkdir(save_directory)
    else:
        index = len(os.listdir(save_directory))

    if n_scenarios == "max" or n_scenarios > len(scenarios):
        n_scenarios = len(scenarios)

    for scenario in scenarios[index:n_scenarios]:
        obs = list()
        actions = list()
        rewards = list()
        terminals = list()
        print(f"Processing scenario {scenario}.")
        vehicle_ids = list()

        scenario_path = Path(input_path) / scenario
        for filename in os.listdir(scenario_path):
            if filename.endswith(".pkl"):
                match = re.search("vehicle-(.*).pkl", filename)
                assert match is not None
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        if len(vehicle_ids) < 2:
            continue

        if n_vehicles == "max" or n_vehicles > len(vehicle_ids):
            n_vehicles = len(vehicle_ids)

        for id in vehicle_ids[0:n_vehicles]:
            print(f"Adding data for vehicle id {id} in scenario {scenario}.")

            with open(
                scenario_path / (f"Agent-history-vehicle-{id}.pkl"),
                "rb",
            ) as f:
                vehicle_data = pickle.load(f)
            image_names = list()

            for filename in os.listdir(scenario_path):
                if filename.endswith(f"-{id}.png"):
                    image_names.append(filename)

            image_names = sorted(image_names)

            goal_pos_x = vehicle_data[float(image_names[-1].split("_Agent")[0])]["ego"][
                "pos"
            ][0]
            goal_pos_y = vehicle_data[float(image_names[-1].split("_Agent")[0])]["ego"][
                "pos"
            ][1]
            threshold = 3

            for i in range(len(image_names) - 1):
                with Image.open(scenario_path / image_names[i], "r") as image:
                    image.seek(0)
                    sim_time = image_names[i].split("_Agent")[0]
                    sim_time_next = image_names[i + 1].split("_Agent")[0]
                    current_position = vehicle_data[float(sim_time)]["ego"]["pos"]
                    current_heading = vehicle_data[float(sim_time)]["ego"]["heading"]
                    next_position = vehicle_data[float(sim_time_next)]["ego"]["pos"]
                    next_heading = vehicle_data[float(sim_time_next)]["ego"]["heading"]
                    trans_coor = get_trans_coor(
                        next_position[0],
                        next_position[1],
                        current_position[0],
                        current_position[1],
                        current_heading,
                    )
                    trans_cur = trans_coor[0]
                    trans_next = trans_coor[1]
                    dx = trans_next[0, 0] - trans_cur[0, 0]
                    dy = trans_next[1, 0] - trans_cur[1, 0]
                    dheading = next_heading - current_heading
                    events = vehicle_data[float(sim_time)]["events"]
                    if all(value == 0 for value in events.values()):
                        terminal = 0
                    else:
                        terminal = 1

                    bev = np.moveaxis(np.asarray(image), -1, 0)
                    goal_obs = get_goal_layer(
                        goal_pos_x,
                        goal_pos_y,
                        current_position[0],
                        current_position[1],
                        current_heading,
                    )
                    extended_ob = np.concatenate((bev, goal_obs), axis=0)
                    obs.append(extended_ob)
                    actions.append([dx, dy, dheading])
                    dist_reward = vehicle_data[float(sim_time)]["dist"]
                    goal_reward = goal_region_reward(
                        threshold,
                        goal_pos_x,
                        goal_pos_y,
                        current_position[0],
                        current_position[1],
                    )
                    rewards.append(dist_reward + goal_reward)
                    rewards.append(dist_reward)

                    terminals.append(terminal)

            print(str(len(obs)) + " pieces of data are added into dataset.")
            n_vehicles = train_config[
                "n_vehicles"
            ]  # Return to default value for next scenario processing

        obs = np.array(obs, dtype=np.uint8)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        dataset = MDPDataset(obs, actions, rewards, terminals)

        if index == 0:
            minimum = [-0.1, 0, -0.1]
            maximum = [0.1, 2, 0.1]
            action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)
            model = d3rlpy.algos.CQL(
                use_gpu=gpu, batch_size=1, action_scaler=action_scaler
            )
        else:
            saved_models = glob.glob(str(save_directory / "*"))
            latest_model = max(saved_models, key=os.path.getctime)
            model = CQL.from_json(
                str(save_directory / "1" / "params.json"), use_gpu=gpu
            )
            model_name = [
                model_name
                for model_name in os.listdir(save_directory / latest_model)
                if model_name.endswith("pt")
            ][0]
            model.load_model(save_directory / latest_model / model_name)
        model.fit(
            dataset,
            eval_episodes=dataset,
            n_steps_per_epoch=n_steps_per_epoch,
            n_steps=n_steps,
            logdir=save_directory,
        )
        saved_models = glob.glob(str(save_directory / "*"))
        latest_model = max(saved_models, key=os.path.getctime)
        os.rename(latest_model, str(save_directory / f"{index + 1}"))
        index += 1

    shutil.rmtree(save_directory)
    model.save_policy(os.path.join(output_path, "model.pt"))


def main(args: argparse.Namespace):
    input_path = args.input_dir
    output_path = args.output_dir
    train(input_path, output_path)


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the offline training data",
        type=str,
        default="/SMARTS/competition/offline_dataset/",
    )
    parser.add_argument(
        "--output_dir",
        help="The path to the directory storing the trained model",
        type=str,
        default="/SMARTS/competition/track2/submission/",
    )

    args = parser.parse_args()

    main(args)
